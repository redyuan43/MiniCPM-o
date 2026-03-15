"""Backend adapters for local duplex inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from local_duplex.config import DuplexRuntimeConfig
from local_duplex.gguf_worker_client import GgufWorkerClient
from local_duplex.vendor import import_duplex_core


LOGGER = logging.getLogger("local_duplex.backend")


class BaseDuplexBackend:
    """Minimal backend interface used by the local runtime loop."""

    backend_name = "unknown"
    observed_logger_names: tuple[str, ...] = ()

    def __init__(self, config: DuplexRuntimeConfig, mode: str) -> None:
        self.config = config
        self.mode = mode

    def preflight(self) -> None:
        """Run backend-specific readiness checks before the session starts."""

    def load(self) -> None:
        raise NotImplementedError

    def prepare(self, system_prompt_text: str, prompt_wav_path: str) -> None:
        raise NotImplementedError

    def prefill(self, audio_waveform, frame_list, max_slice_nums: int) -> None:
        raise NotImplementedError

    def generate(self, listen_prob_scale_override: float | None = None):
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def set_break(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError

    def is_vision_oom(self, exc: Exception) -> bool:
        return isinstance(exc, torch.OutOfMemoryError)


class PytorchDuplexBackend(BaseDuplexBackend):
    """Current UnifiedProcessor-based duplex backend."""

    backend_name = "pytorch"
    observed_logger_names = (
        "MiniCPMO45.utils",
        "MiniCPMO45.modeling_minicpmo_unified",
    )

    def __init__(self, config: DuplexRuntimeConfig, mode: str) -> None:
        super().__init__(config, mode)
        self._processor = None
        self._duplex = None

    def load(self) -> None:
        UnifiedProcessor, DuplexConfig = import_duplex_core()
        from MiniCPMO45.utils import DuplexWindowConfig

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        processor = UnifiedProcessor(
            model_path=self.config.model.model_path,
            device=self.config.model.device,
            ref_audio_path=self.config.audio.reference_audio_path,
            duplex_config=DuplexConfig(
                chunk_ms=self.config.audio.chunk_ms,
                force_listen_count=self.config.session.force_listen_count,
                max_new_speak_tokens_per_chunk=self.config.session.max_new_speak_tokens_per_chunk,
                temperature=self.config.session.temperature,
                top_k=self.config.session.top_k,
                top_p=self.config.session.top_p,
                listen_prob_scale=self.config.session.listen_prob_scale,
            ),
            preload_both_tts=self.config.model.preload_both_tts,
            compile=self.config.model.compile,
            attn_implementation=self.config.model.attn_implementation,
        )
        self._processor = processor
        processor.ref_audio_path = None
        self._duplex = processor.set_duplex_mode()
        self._duplex.ref_audio_path = None
        self._duplex._model.duplex.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode="basic",
                basic_window_high_tokens=1800,
                basic_window_low_tokens=1400,
                context_previous_max_tokens=256,
                context_max_units=12,
            )
        )
        self._duplex._model.duplex.decoder.set_window_enabled(True)

    def prepare(self, system_prompt_text: str, prompt_wav_path: str) -> None:
        assert self._duplex is not None
        self._duplex.prepare(
            system_prompt_text=system_prompt_text,
            prompt_wav_path=prompt_wav_path,
        )

    def prefill(self, audio_waveform, frame_list, max_slice_nums: int) -> None:
        assert self._duplex is not None
        self._duplex.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def generate(self, listen_prob_scale_override: float | None = None):
        assert self._duplex is not None
        original = self._duplex.config.listen_prob_scale
        if listen_prob_scale_override is not None:
            self._duplex.config.listen_prob_scale = listen_prob_scale_override
        try:
            return self._duplex.generate()
        finally:
            self._duplex.config.listen_prob_scale = original

    def finalize(self) -> None:
        assert self._duplex is not None
        self._duplex.finalize()

    def set_break(self) -> None:
        assert self._duplex is not None
        self._duplex.set_break()

    def stop(self) -> None:
        if self._duplex is not None:
            self._duplex.stop()

    def cleanup(self) -> None:
        if self._duplex is not None:
            self._duplex.cleanup()


class GgufDuplexBackend(BaseDuplexBackend):
    """GGUF backend for local non-HTTP llama.cpp-omni integration."""

    backend_name = "gguf"
    observed_logger_names = ("local_duplex.gguf_worker",)

    def __init__(self, config: DuplexRuntimeConfig, mode: str) -> None:
        super().__init__(config, mode)
        self._client: GgufWorkerClient | None = None

    def preflight(self) -> None:
        missing_paths = [
            path
            for path in (
                self.config.model.gguf_model_path,
                self.config.model.gguf_audio_path,
                self.config.model.gguf_vision_path,
                self.config.model.gguf_tts_path,
                self.config.model.gguf_projector_path,
                self.config.model.gguf_token2wav_dir,
            )
            if not Path(path).exists()
        ]
        if missing_paths:
            joined = "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(
                "GGUF 本地后端缺少必需文件，请先运行 scripts/prepare_local_duplex_gguf.sh。\n"
                f"缺失路径:\n{joined}"
            )
        worker_root = Path(self.config.model.gguf_worker_root)
        if not worker_root.exists():
            raise FileNotFoundError(
                "GGUF 推理框架目录不存在，请先运行 scripts/build_local_duplex_gguf_backend.sh。\n"
                f"缺失目录: {worker_root}"
            )
        worker_bin = Path(self.config.model.gguf_worker_bin)
        if not worker_bin.exists():
            raise FileNotFoundError(
                "GGUF 推理二进制不存在，请先运行 scripts/build_local_duplex_gguf_backend.sh。\n"
                f"缺失文件: {worker_bin}"
            )

    def load(self) -> None:
        self._client = GgufWorkerClient(
            config=self.config,
            mode=self.mode,
            logger_name=self.observed_logger_names[0],
        )
        self._client.start()

    def prepare(self, system_prompt_text: str, prompt_wav_path: str) -> None:
        assert self._client is not None
        self._client.prepare(
            system_prompt_text=system_prompt_text,
            prompt_wav_path=prompt_wav_path,
        )

    def prefill(self, audio_waveform, frame_list, max_slice_nums: int) -> None:
        assert self._client is not None
        self._client.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def generate(self, listen_prob_scale_override: float | None = None):
        assert self._client is not None
        return self._client.generate(listen_prob_scale_override)

    def finalize(self) -> None:
        return None

    def set_break(self) -> None:
        assert self._client is not None
        self._client.set_break()

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()

    def cleanup(self) -> None:
        if self._client is not None:
            self._client.cleanup()

    def is_vision_oom(self, exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()


def create_duplex_backend(config: DuplexRuntimeConfig, mode: str) -> BaseDuplexBackend:
    backend = config.model.backend.lower().strip()
    if backend == "pytorch":
        return PytorchDuplexBackend(config, mode)
    if backend == "gguf":
        return GgufDuplexBackend(config, mode)
    raise ValueError(f"Unsupported local duplex backend: {config.model.backend}")
