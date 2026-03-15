#include "common.h"
#include "omni.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct WorkerState {
    common_params params;
    omni_context * ctx = nullptr;
    std::string mode = "audio";
    std::string base_output_dir;
    std::string prompt_wav_path;
    int chunk_index = 1;
    int last_seen_wav_index = -1;
    int wav_wait_ms = 900;
    int wav_idle_stable_ms = 80;
    int wav_empty_wait_ms = 220;
    int trailing_wait_ms = 150;
    int trailing_idle_stable_ms = 60;
    bool initialized = false;
    bool prepared = false;
};

struct WavWaitResult {
    std::vector<std::string> paths;
    int waited_ms = 0;
};

struct WavCollectResult {
    std::vector<std::string> paths;
    int wav_wait_ms = 0;
    int trailing_wait_ms = 0;
};

std::optional<int> parse_wav_index(const fs::path & path) {
    const std::string name = path.filename().string();
    if (name.size() < 9 || name.rfind("wav_", 0) != 0 || name.substr(name.size() - 4) != ".wav") {
        return std::nullopt;
    }
    try {
        return std::stoi(name.substr(4, name.size() - 8));
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<std::pair<int, std::string>> scan_new_wavs(const std::string & base_output_dir, int last_seen_wav_index) {
    std::vector<std::pair<int, std::string>> wavs;
    const fs::path wav_dir = fs::path(base_output_dir) / "tts_wav";
    if (!fs::exists(wav_dir)) {
        return wavs;
    }
    for (const auto & entry : fs::directory_iterator(wav_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto idx = parse_wav_index(entry.path());
        if (!idx.has_value() || idx.value() <= last_seen_wav_index) {
            continue;
        }
        wavs.emplace_back(idx.value(), entry.path().string());
    }
    std::sort(wavs.begin(), wavs.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.first < rhs.first;
    });
    return wavs;
}

int scan_highest_wav_index(const std::string & base_output_dir) {
    int highest = -1;
    for (const auto & item : scan_new_wavs(base_output_dir, -1)) {
        highest = std::max(highest, item.first);
    }
    return highest;
}

WavWaitResult wait_for_new_wavs(
    const std::string & base_output_dir,
    int & last_seen_wav_index,
    int max_wait_ms,
    int idle_stable_ms,
    int empty_wait_ms
) {
    constexpr int kPollMs = 20;

    auto best = scan_new_wavs(base_output_dir, last_seen_wav_index);
    auto start = std::chrono::steady_clock::now();
    auto last_change = start;

    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() < max_wait_ms) {
        auto current = scan_new_wavs(base_output_dir, last_seen_wav_index);
        if (current.size() != best.size()) {
            best = std::move(current);
            last_change = std::chrono::steady_clock::now();
        } else if (!current.empty() && !best.empty() && current.back().first != best.back().first) {
            best = std::move(current);
            last_change = std::chrono::steady_clock::now();
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();
        if (best.empty() && elapsed_ms >= empty_wait_ms) {
            break;
        }
        if (!best.empty()) {
            const auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_change
            ).count();
            if (idle_ms >= idle_stable_ms) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
    }

    std::vector<std::string> paths;
    for (const auto & item : best) {
        last_seen_wav_index = std::max(last_seen_wav_index, item.first);
        paths.push_back(item.second);
    }
    const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();
    return {
        std::move(paths),
        static_cast<int>(waited_ms),
    };
}

WavCollectResult collect_generate_wavs(
    WorkerState & state,
    bool end_of_turn
) {
    auto wav_result = wait_for_new_wavs(
        state.base_output_dir,
        state.last_seen_wav_index,
        state.wav_wait_ms,
        state.wav_idle_stable_ms,
        state.wav_empty_wait_ms
    );
    if (!end_of_turn || wav_result.paths.empty()) {
        return {
            std::move(wav_result.paths),
            wav_result.waited_ms,
            0,
        };
    }

    // The final token2wav flush can land slightly after the normal stable window.
    auto trailing_result = wait_for_new_wavs(
        state.base_output_dir,
        state.last_seen_wav_index,
        state.trailing_wait_ms,
        state.trailing_idle_stable_ms,
        state.trailing_wait_ms
    );
    wav_result.paths.insert(
        wav_result.paths.end(),
        trailing_result.paths.begin(),
        trailing_result.paths.end()
    );
    return {
        std::move(wav_result.paths),
        wav_result.waited_ms,
        trailing_result.waited_ms,
    };
}

void drain_text_queue(omni_context * ctx, bool & is_listen, bool & end_of_turn, std::string & text) {
    std::lock_guard<std::mutex> lock(ctx->text_mtx);
    while (!ctx->text_queue.empty()) {
        const std::string item = ctx->text_queue.front();
        ctx->text_queue.pop_front();
        if (item == "__IS_LISTEN__") {
            is_listen = true;
            continue;
        }
        if (item == "__END_OF_TURN__") {
            end_of_turn = true;
            continue;
        }
        text += item;
    }
}

void stop_and_free(WorkerState & state) {
    if (state.ctx == nullptr) {
        return;
    }
    omni_stop_threads(state.ctx);
    if (state.ctx->llm_thread.joinable()) {
        state.ctx->llm_thread.join();
    }
    if (state.ctx->use_tts && state.ctx->tts_thread.joinable()) {
        state.ctx->tts_thread.join();
    }
    if (state.ctx->use_tts && state.ctx->t2w_thread.joinable()) {
        state.ctx->t2w_thread.join();
    }
    omni_free(state.ctx);
    state.ctx = nullptr;
    state.initialized = false;
    state.prepared = false;
}

void write_protocol(FILE * out, const json & payload) {
    const std::string line = payload.dump();
    std::fwrite(line.data(), 1, line.size(), out);
    std::fwrite("\n", 1, 1, out);
    std::fflush(out);
}

void apply_duplex_prompts(omni_context * ctx, const std::string & system_prompt_text) {
    const std::string prefix = "<|im_start|>system\n" + system_prompt_text + "\n<|audio_start|>";
    const std::string suffix = "<|audio_end|><|im_end|>\n";
    ctx->audio_voice_clone_prompt = prefix;
    ctx->audio_assistant_prompt = suffix;
    ctx->omni_voice_clone_prompt = prefix;
    ctx->omni_assistant_prompt = suffix;
}

json handle_init(WorkerState & state, const json & req) {
    stop_and_free(state);

    state.mode = req.value("mode", "audio");
    state.base_output_dir = req.at("base_output_dir").get<std::string>();
    fs::create_directories(fs::path(state.base_output_dir) / "tts_wav");

    state.params = common_params();
    state.params.model.path = req.at("llm_model_path").get<std::string>();
    state.params.apm_model = req.at("audio_model_path").get<std::string>();
    state.params.vpm_model = req.at("vision_model_path").get<std::string>();
    state.params.tts_model = req.at("tts_model_path").get<std::string>();
    state.params.n_ctx = req.value("ctx_size", 4096);
    state.params.n_predict = req.value("n_predict", 256);
    state.params.n_gpu_layers = req.value("n_gpu_layers", -1);
    state.params.main_gpu = req.value("device_index", 0);
    state.params.sampling.temp = req.value("temperature", 0.7f);
    state.params.sampling.top_k = req.value("top_k", 20);
    state.params.sampling.top_p = req.value("top_p", 0.8f);
    state.params.sampling.min_p = 0.0f;
    state.params.cpuparams.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    state.params.cpuparams_batch.n_threads = state.params.cpuparams.n_threads;

    const std::string tts_model_path = req.at("tts_model_path").get<std::string>();
    const std::string token2wav_dir = req.at("token2wav_dir").get<std::string>();
    const std::string token2wav_device = "gpu:0";
    const int media_type = state.mode == "omni" ? 2 : 1;
    const std::string tts_bin_dir = fs::path(tts_model_path).parent_path().string();

    common_init();

    state.ctx = omni_init(
        &state.params,
        media_type,
        req.value("use_tts", true),
        tts_bin_dir,
        -1,
        token2wav_device,
        true,
        nullptr,
        nullptr,
        state.base_output_dir
    );
    if (state.ctx == nullptr) {
        throw std::runtime_error("omni_init failed");
    }
    state.ctx->async = true;
    state.ctx->listen_prob_scale = req.value("listen_prob_scale", 1.0f);
    state.ctx->max_new_speak_tokens_per_chunk = req.value("max_new_speak_tokens_per_chunk", 24);
    state.ctx->token2wav_model_dir = token2wav_dir;
    state.ctx->clean_kvcache = true;
    state.wav_wait_ms = req.value("wav_wait_ms", 900);
    state.wav_idle_stable_ms = req.value("wav_idle_stable_ms", 80);
    state.wav_empty_wait_ms = req.value("wav_empty_wait_ms", 220);
    state.trailing_wait_ms = req.value("trailing_wait_ms", 150);
    state.trailing_idle_stable_ms = req.value("trailing_idle_stable_ms", 60);
    state.chunk_index = 1;
    state.last_seen_wav_index = scan_highest_wav_index(state.base_output_dir);
    state.initialized = true;
    state.prepared = false;

    return {
        {"ok", true},
        {"message", "initialized"},
    };
}

json handle_prepare(WorkerState & state, const json & req) {
    if (!state.initialized || state.ctx == nullptr) {
        throw std::runtime_error("worker is not initialized");
    }
    state.prompt_wav_path = req.at("prompt_wav_path").get<std::string>();
    stop_speek(state.ctx);
    clean_kvcache(state.ctx);
    state.ctx->system_prompt_initialized = false;
    state.ctx->current_turn_ended = true;
    state.ctx->ended_with_listen = false;
    state.ctx->break_event.store(false);
    state.ctx->ref_audio_path = state.prompt_wav_path;
    apply_duplex_prompts(state.ctx, req.at("system_prompt_text").get<std::string>());

    const bool ok = stream_prefill(state.ctx, state.prompt_wav_path, "", 0, -1);
    if (!ok) {
        throw std::runtime_error("stream_prefill failed during prepare");
    }
    state.chunk_index = 1;
    state.last_seen_wav_index = scan_highest_wav_index(state.base_output_dir);
    state.prepared = true;
    return {
        {"ok", true},
        {"message", "prepared"},
    };
}

json handle_prefill(WorkerState & state, const json & req) {
    if (!state.prepared || state.ctx == nullptr) {
        throw std::runtime_error("worker is not prepared");
    }
    const std::string audio_path = req.at("audio_path").get<std::string>();
    const std::string image_path = req.value("image_path", "");
    const int max_slice_nums = req.value("max_slice_nums", -1);
    const bool ok = stream_prefill(state.ctx, audio_path, image_path, state.chunk_index, max_slice_nums);
    if (!ok) {
        throw std::runtime_error("stream_prefill failed");
    }
    return {
        {"ok", true},
        {"message", "prefilled"},
        {"chunk_index", state.chunk_index},
    };
}

json handle_generate(WorkerState & state, const json & req) {
    if (!state.prepared || state.ctx == nullptr) {
        throw std::runtime_error("worker is not prepared");
    }
    state.ctx->listen_prob_scale = req.value("listen_prob_scale", state.ctx->listen_prob_scale);
    const auto started = std::chrono::steady_clock::now();
    const bool ok = stream_decode(state.ctx, (fs::path(state.base_output_dir) / "llm_debug").string(), -1);
    const auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started
    ).count();
    if (!ok) {
        throw std::runtime_error("stream_decode failed");
    }

    bool is_listen = false;
    bool end_of_turn = false;
    std::string text;
    drain_text_queue(state.ctx, is_listen, end_of_turn, text);
    const bool should_collect_wavs =
        !is_listen || end_of_turn || !text.empty();
    const auto wav_result = should_collect_wavs
        ? collect_generate_wavs(state, end_of_turn)
        : WavCollectResult{};
    const int used_chunk_index = state.chunk_index;
    state.chunk_index += 1;
    const auto cost_all_ms = decode_ms + wav_result.wav_wait_ms + wav_result.trailing_wait_ms;

    return {
        {"ok", true},
        {"chunk_index", used_chunk_index},
        {"is_listen", is_listen},
        {"end_of_turn", end_of_turn},
        {"text", text},
        {"audio_wav_paths", wav_result.paths},
        {"decode_ms", decode_ms},
        {"wav_wait_ms", wav_result.wav_wait_ms},
        {"trailing_wait_ms", wav_result.trailing_wait_ms},
        {"cost_all_ms", cost_all_ms},
        {"n_tokens", nullptr},
        {"n_tts_tokens", nullptr},
    };
}

json handle_break(WorkerState & state) {
    if (!state.initialized || state.ctx == nullptr) {
        throw std::runtime_error("worker is not initialized");
    }
    state.ctx->break_event.store(true);
    stop_speek(state.ctx);
    return {
        {"ok", true},
        {"message", "break_set"},
    };
}

}  // namespace

int main() {
    const int protocol_fd = dup(STDOUT_FILENO);
    if (protocol_fd < 0) {
        return 1;
    }
    if (dup2(STDERR_FILENO, STDOUT_FILENO) == -1) {
        return 1;
    }
    FILE * protocol_out = fdopen(protocol_fd, "w");
    if (protocol_out == nullptr) {
        return 1;
    }
    std::setvbuf(protocol_out, nullptr, _IOLBF, 0);

    WorkerState state;
    std::string line;
    while (std::getline(std::cin, line)) {
        json response;
        int seq = -1;
        try {
            const json req = json::parse(line);
            seq = req.value("seq", -1);
            const std::string type = req.at("type").get<std::string>();
            if (type == "init") {
                response = handle_init(state, req);
            } else if (type == "prepare") {
                response = handle_prepare(state, req);
            } else if (type == "prefill") {
                response = handle_prefill(state, req);
            } else if (type == "generate") {
                response = handle_generate(state, req);
            } else if (type == "break") {
                response = handle_break(state);
            } else if (type == "shutdown") {
                stop_and_free(state);
                response = {{"ok", true}, {"message", "shutdown"}};
                response["seq"] = seq;
                write_protocol(protocol_out, response);
                break;
            } else {
                throw std::runtime_error("unknown command type: " + type);
            }
            response["seq"] = seq;
        } catch (const std::exception & exc) {
            response = {
                {"ok", false},
                {"seq", seq},
                {"error", exc.what()},
            };
        }
        write_protocol(protocol_out, response);
    }

    stop_and_free(state);
    std::fclose(protocol_out);
    return 0;
}
