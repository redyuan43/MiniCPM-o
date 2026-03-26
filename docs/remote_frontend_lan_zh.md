# MiniCPM-o 局域网远端前端部署

本文档对应以下运行形态：

- 后端机器：只负责 GPU 推理，运行官方 `gateway + worker`
- 远端设备：通过浏览器打开前端页面，负责
  - 麦克风输入
  - 摄像头输入
  - 喇叭/耳机输出

这正是官方 Web Demo 的原生架构，不需要使用 `local_duplex`。

## 架构

```text
Remote Browser (mic + camera + speaker)
        |
        | HTTPS / WSS
        v
MiniCPM-o Gateway (LAN host)
        |
        | HTTP / WS (internal only)
        v
MiniCPM-o Worker (GPU)
```

推荐页面：

- `/omni`：视频 + 语音全双工
- `/audio_duplex`：纯语音全双工
- `/half_duplex`：半双工作为稳定基线

## 前提

- 后端机器已经准备好模型、Python 环境和依赖
- 后端机器和远端设备在同一个局域网
- 后端机器有固定内网 IP 或固定主机名
- 后端机器的防火墙允许外部访问 Gateway 端口

如果你还没完成后端环境准备，先运行：

```bash
bash scripts/bootstrap_minicpmo45_pytorch_demo_spark.sh
```

## 第一步：确认后端访问地址

先在后端机器上确认局域网地址，例如：

```bash
hostname -I
```

假设后端机器 IP 是 `192.168.1.23`。

也可以给它一个局域网主机名，例如 `minicpmo.lan`。如果远端设备能正确解析主机名，优先使用主机名。

## 第二步：启动后端服务

### 方案 A：直接用匹配局域网地址的自签证书

适合先跑通链路。

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_minicpmo45_pytorch_demo_spark.sh \
  --public-host 192.168.1.23 \
  --regen-cert
```

说明：

- `--public-host` 是远端浏览器实际访问的地址
- `--regen-cert` 会重新生成 SAN 覆盖该地址的自签证书
- Gateway 默认绑定 `0.0.0.0`，局域网设备可直接访问

如果你使用局域网主机名：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_minicpmo45_pytorch_demo_spark.sh \
  --public-host minicpmo.lan \
  --tls-extra-ip 192.168.1.23 \
  --regen-cert
```

### 方案 B：使用你自己的证书

适合正式长期使用。

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_minicpmo45_pytorch_demo_spark.sh \
  --public-host minicpmo.lan \
  --ssl-certfile /abs/path/minicpmo.lan.pem \
  --ssl-keyfile /abs/path/minicpmo.lan-key.pem
```

要求：

- 证书的 SAN 必须覆盖远端浏览器访问时用到的主机名或 IP
- 推荐使用受信任证书；如果是自签证书，远端设备需要手动信任

## 第三步：远端设备访问前端

在远端设备浏览器中打开：

```text
https://192.168.1.23:18006/omni
https://192.168.1.23:18006/audio_duplex
https://192.168.1.23:18006/half_duplex
```

如果你配置的是主机名，则把 IP 换成主机名。

远端设备需要：

- 允许浏览器使用麦克风
- `omni` 页面额外允许浏览器使用摄像头
- 使用耳机时，全双工打断成功率通常更高

## 第四步：建议的联调顺序

先按下面顺序排查，最省时间：

1. 打开 `/audio_duplex`
2. 确认浏览器能拿到麦克风权限
3. 确认模型能返回语音
4. 再打开 `/omni`
5. 确认浏览器能拿到摄像头权限并看到本机视频预览

这样可以先把 TLS、WSS、麦克风、喇叭这些链路跑通，再加视频。

## 健康检查

后端机器本地：

```bash
bash scripts/smoke_minicpmo45_pytorch_demo.sh
```

远端设备浏览器或命令行可访问：

```text
https://192.168.1.23:18006/health
https://192.168.1.23:18006/status
```

## 防火墙建议

只对局域网开放 Gateway 端口，例如 `18006`。

不要把 Worker 端口 `22400+` 暴露给其他设备；它们只需要本机 `Gateway -> Worker` 通信。

## 常见问题

### 1. 页面能打开，但麦克风/摄像头权限拿不到

优先检查：

- 远端浏览器访问的 URL 是否是 `https://...`
- 证书是否覆盖你访问时使用的主机名或 IP
- 浏览器是否把当前站点视为安全上下文

### 2. 后端本机可打开，远端设备打不开

优先检查：

- `gateway.py` 是否监听在 `0.0.0.0`
- 后端机器防火墙是否放通 Gateway 端口
- 远端设备是否和后端在同一网段

### 3. 远端访问时提示证书不匹配

说明当前证书 SAN 不包含你实际访问的地址。重新生成：

```bash
bash scripts/start_minicpmo45_pytorch_demo_spark.sh \
  --public-host 192.168.1.23 \
  --regen-cert
```

### 4. 全双工时容易回声或打断不稳定

这是全双工常见现象。建议：

- 远端设备优先使用耳机
- 先验证 `/audio_duplex`
- 再验证 `/omni`

## 停止服务

```bash
bash scripts/stop_minicpmo45_pytorch_demo_spark.sh
```
