# MiniCPM-o 本地全双工落地总结

## 1. 目标

本次工作的目标不是 Web Demo，也不是 HTTP / WebRTC，而是把 `MiniCPM-o` 改造成一套适合机器人后续接入的本地端到端运行方式：

- 输入：
  - 本地麦克风
  - 本地摄像头
- 输出：
  - 本地扬声器
- 推理：
  - 本地 GPU
- 目标形态：
  - 全双工
  - 端到端
  - 无浏览器依赖
  - 无 HTTP 依赖

## 2. 硬件和系统前提

- 操作系统：Ubuntu 24
- GPU：
  - RTX 4060 Ti 16GB
  - Tesla V100 16GB
- 最终选型：
  - 主推理 GPU 使用 `RTX 4060 Ti`
  - 不使用 `V100` 作为主路径
- 外设：
  - 麦克风：`DJI MIC MINI`
  - 摄像头：`Logitech BRIO`
  - 扬声器：`NVIDIA HDMI/DisplayPort-HDA`

经验：

- 对当前目标来说，`4060 Ti` 比 `V100` 更合适，因为我们优先保证本地实时交互能力和兼容性。
- HDMI 输出设备只支持固定采样率和双声道，因此播放链路必须显式做格式转换，不能把格式转换全交给 ALSA 黑盒处理。

## 3. 路线选择

最开始参考了 `MiniCPM-o` 主仓库 README，但最终没有走浏览器/WebRTC 路径，而是落到本地运行时：

- 主仓库：`OpenBMB/MiniCPM-o`
- 真正使用的本地 duplex 核心：`OpenBMB/MiniCPM-o-Demo`

原因：

- 你的目标是后续赋能机器人，不是先搭建网页端产品。
- 本地设备输入输出更接近未来机器人硬件接入方式。
- `MiniCPM-o-Demo` 已经提供了 duplex 的核心调用链：
  - `prepare()`
  - `prefill()`
  - `generate()`
  - `finalize()`

## 4. 环境安装步骤

### 4.1 系统依赖

需要准备这些系统工具和库：

- `uv`
- `git`
- `ffmpeg`
- `cmake`
- `ninja-build`
- `pkg-config`
- `libasound2-dev`
- `v4l-utils`

### 4.2 Python 虚拟环境

创建了独立运行环境：

- 路径：`.venv/local-duplex311`
- Python 版本：`3.11`

关键原因：

- 上游运行时和依赖在 Python 3.11 上更稳
- Ubuntu 24 默认环境不适合直接裸跑这一套依赖

执行脚本：

```bash
./scripts/setup_duplex_env.sh
```

这个脚本会做以下事情：

1. 创建 `Python 3.11` venv
2. 安装 `CUDA 12.4` 对应的 PyTorch
3. 安装 runtime 依赖
4. 准备本地 duplex 所需 Python 包

### 4.3 关键 Python 依赖

实际安装中比较关键的依赖包括：

- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `transformers==4.51.0`
- `accelerate`
- `autoawq`
- `opencv-python`
- `librosa`
- `soundfile`
- `decord`
- `onnxruntime`
- `minicpmo-utils`

## 5. 模型和代码组织

### 5.1 运行时代码

新增了本地运行时目录：

- `local_duplex/`

它负责：

- 本地音频采集
- 本地摄像头采集
- duplex 会话循环
- ALSA 播放
- 本地 CLI

### 5.2 参考上游代码

引入了：

- `third_party/MiniCPM-o-Demo`

但不是原样使用，而是做了本地定制和补丁。

### 5.3 模型本地化

为了让模型不依赖用户目录下的 Hugging Face 缓存，并且便于打包，最终把模型缓存迁移到当前工程内：

- `third_party/models/huggingface/hub/models--openbmb--MiniCPM-o-4_5-awq`
- `third_party/models/huggingface/hub/models--openbmb--MiniCPM-o-4_5`

这样做的原因：

- 当前工程可自包含
- 后续打包、迁移、部署更容易
- 同时保留 Hugging Face 的 cache 结构，避免破坏上游的加载逻辑

执行脚本：

```bash
./scripts/localize_models.sh
```

## 6. 输入输出格式设计

这部分是为了实时性专门做的。

### 6.1 输入音频

- 麦克风输入格式：`16 kHz`、`mono`、`float32`
- 这是直接喂给模型的格式
- 尽量避免中间重复重采样

### 6.2 输出音频

- 模型输出格式：`24 kHz`、`mono`、`float32`
- NVIDIA HDMI 设备实际播放格式：`48 kHz`、`stereo`

因此运行时显式做了：

- `24k mono float32 -> 48k stereo`

经验：

- 不要把这个格式转换完全交给 ALSA 自动处理
- 显式转换更稳定，也更可控

### 6.3 视频输入

- 摄像头：`Logitech BRIO`
- 采样：`MJPEG`
- 默认运行时使用较轻的参数来平衡实时性
- 模型不是每帧都吃，而是按节奏取最新画面

经验：

- 本地实时交互不应该积压视频帧
- 永远只保留“最新帧”更适合机器人实时场景

## 7. 关键问题和修复经验

### 7.1 `pkg_resources` 缺失

问题：

- `librosa==0.9.0` 依赖 `pkg_resources`
- 新版 `setuptools` 不再稳定提供这个接口

解决：

- 固定 `setuptools<81`

经验：

- 老音频栈和新打包工具经常有兼容问题
- 对这类项目不要盲目追最新 setuptools

### 7.2 AWQ 相关依赖缺失

问题：

- AWQ 模型加载需要 `autoawq`

解决：

- 补装 `autoawq>=0.1.8`

### 7.3 HDMI 播放格式不匹配

问题：

- NVIDIA HDMI 设备不是随便什么格式都能直接播

解决：

- 运行时显式转成 `S32_LE`
- 再通过 `aplay` 输出

经验：

- 对 HDMI/DP 音频设备，先相信设备能力，再设计播放链路

### 7.4 4060 显存紧张

问题：

- 4060 只有 16GB
- 运行 unified + AWQ + TTS + vision 时显存比较紧

解决：

- token2wav 走 `fp16`
- 避免不必要的 GPU cache 常驻
- 视觉 OOM 时允许降级策略
- 清理无关 GPU 进程

经验：

- 真正阻塞往往不是模型本身，而是后台常驻 GPU 进程
- 每次验证前都应该先看 `nvidia-smi`

### 7.5 Duplex 会话状态漂移

问题：

- 长时间运行后会出现：
  - `CONSISTENCY ERROR`
  - `audio_past_key_values length exceed`
- 然后模型容易长时间卡在 `listen`

解决：

- 在本地 runtime 增加 session reset 策略
- 当：
  - 单个会话 chunk 太多
  - 或检测到讲话但模型连续保持 `listen`
  - 就自动重建 duplex session

经验：

- 这类全双工状态机不能无限相信上游内部缓存
- 工程层必须准备恢复机制

## 8. 最终验证结果

最终已经验证到：

- 可以在 `RTX 4060 Ti` 上加载模型
- 可以进入本地 omni duplex 运行态
- 可以读取摄像头画面
- 可以读取麦克风输入
- 可以跑通本地全双工会话循环
- 日志中可以看到：
  - `has_frames=True`
  - `has_audio=True`
  - `speech_detected=True`

结论：

- “看得到” 已确认
- “听得到” 已确认
- 本地 speaker 链路已打通
- 运行时已具备继续做机器人接入的基础

## 9. 当前工程里的常用命令

### 环境安装

```bash
./scripts/setup_duplex_env.sh
```

### 模型本地化

```bash
./scripts/localize_models.sh
```

### 运行

```bash
./scripts/run_local_duplex.sh omni
```

### 看日志

```bash
./scripts/logs_local_duplex.sh
```

### 停止

```bash
./scripts/stop_local_duplex.sh
```

### 打包环境和模型

```bash
./scripts/package_local_duplex_bundle.sh
```

## 10. 建议

后续如果要进入机器人接入阶段，建议按这个顺序继续：

1. 保持本地 ALSA / 摄像头版运行稳定
2. 把 `local_duplex` 里的设备适配层抽象出来
3. 用机器人真实音频输入、扬声器输出、摄像头驱动替换 ALSA/OpenCV
4. 保留当前这套本地 CLI 作为回归测试基线

核心经验总结：

- 先把本地端到端打通，再接机器人硬件
- 先保证数据格式最优，再谈实时性
- 先控制 GPU 占用，再谈模型稳定性
- 先增加恢复机制，再追求长时间连续运行
