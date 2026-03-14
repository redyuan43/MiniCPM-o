# Local MiniCPM-o Duplex on Ubuntu 24

This repo now includes a local duplex runtime for MiniCPM-o 4.5 that keeps camera, microphone, model inference, and speaker playback on the same machine.

## Hardware Defaults

- Inference GPU: `GPU 0 / RTX 4060 Ti`
- Microphone: `DJI MIC MINI` via `plughw:1,0`
- Speaker: `NVIDIA HDMI/DP HDA` via `hw:CARD=NVidia,DEV=3`
- Camera: `Logitech BRIO` via `/dev/v4l/by-id/usb-046d_Logitech_BRIO_AE03BDC5-video-index0`

## Runtime Profile

- Audio chunk size: `500 ms`
- Audio input format: `16 kHz mono float32`
- Model output format: `24 kHz mono float32`
- HDMI playback format: explicit app-side conversion to `48 kHz stereo float32`
- Camera capture: `1280x720 MJPEG`
- Vision cadence into the model: one frame every second

## Environment Setup

```bash
./scripts/setup_duplex_env.sh
```

The setup script:

- vendors `third_party/MiniCPM-o-Demo`
- creates `.venv/local-duplex311`
- installs CUDA-enabled PyTorch for Python 3.11
- installs the local runtime requirements
- prepares the workspace for repo-local Hugging Face model storage under `third_party/models/huggingface`

## Model Configuration

The default config file is [configs/local_duplex.json](/home/ivan/github/MiniCPM-o/configs/local_duplex.json).

Default model:

- `openbmb/MiniCPM-o-4_5-awq`

Localize the model cache into the current project before packaging or offline use:

```bash
./scripts/localize_models.sh
```

The launcher now prefers:

- `HF_HOME=$PWD/third_party/models/huggingface`
- `HF_HUB_CACHE=$PWD/third_party/models/huggingface/hub`

If the model is already downloaded locally, set `"model_path"` to that local directory or override it from the CLI:

```bash
.venv/local-duplex311/bin/python -m local_duplex.cli omni --model-path "/path/to/model"
```

## Running

Foreground:

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH="$PWD:$PWD/third_party/MiniCPM-o-Demo" \
.venv/local-duplex311/bin/python -m local_duplex.cli omni --config configs/local_duplex.json
```

Background helper scripts:

```bash
./scripts/start_local_duplex.sh omni
./scripts/logs_local_duplex.sh
./scripts/stop_local_duplex.sh
```

Bundle the runtime, local venv, and localized models:

```bash
./scripts/package_local_duplex_bundle.sh
```

Audio-only mode:

```bash
./scripts/start_local_duplex.sh audio
```

## Notes

- The runtime checks that the active NVIDIA HDMI sink is present before starting.
- The OpenCV preview window is enabled in `omni` mode by default; use `--no-preview` to disable it.
- If you need a different mic, speaker, or camera, override the corresponding fields in the config file or CLI flags.
