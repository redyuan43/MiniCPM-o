# Local Duplex 当前进度

## 当前进度

本地全双工运行时已经从最初的 AWQ/PyTorch 路线，迁移到默认使用 `GGUF Q4_K_M` 的本地后端，并且保留了项目内设备接入、交互日志和会话健康管理能力。

当前建议的测试入口仍然是：

```bash
./scripts/run_local_duplex.sh omni
./scripts/run_local_duplex.sh omni --audio-only
./scripts/logs_local_duplex.sh
./scripts/interaction_log.sh
./scripts/show_local_duplex_session.sh
```

## 已完成任务

### 1. 本地 GGUF 运行时

- 增加了 GGUF 后端抽象与本地 worker 客户端。
- 默认模型切到 `MiniCPM-o 4.5 GGUF / Q4_K_M`。
- 保留 `audio` 和 `omni` 两种模式，并支持 `--audio-only`。

### 2. 输入输出设备链路

- 输入侧已经改为 `sounddevice + PortAudio`，不再依赖不稳定的 `arecord` 独占采集。
- 输出侧新增共享播放后端，默认改为 `sounddevice + pipewire`，不再默认绑定 `aplay -D hw:...`。
- 对于显式指定 `hw:/plughw:/sysdefault:` 的场景，仍保留 `aplay` 兼容路径。

### 3. TTS 输出提速与平滑

- 调整了 GGUF worker 的 wav 收集窗口，降低了无意义等待。
- 增加了播放侧的最小缓冲和 turn 合并逻辑，减少“几个字一卡”的问题。
- 增加了句尾补收集逻辑，避免最后一小段 wav 直接丢失。

### 4. 会话健康与日志

- 增加了 session health 状态和重置原因记录。
- 增加了用户可读交互日志：
  - `interaction.jsonl`
  - `interaction.md`
  - `summary.json`
- 增加了 `interaction_log.sh`，可以直接查看对话时间线和性能摘要。

### 5. 纯音频模式

- 增加了 `--audio-only`，用于只测试音频输入和音频输出链路。
- 从近期日志看，纯音频模式的 `speak latency` 已经明显低于带视觉模式。

## 当前已知问题

以下问题在当前版本中仍然存在：

### 1. `./scripts/run_local_duplex.sh omni --audio-only` 仍然会出现断断续续

- 在部分会话中，虽然 `interaction_log.sh` 显示 `speak latency` 已经低于 `1000ms`，但实际听感仍然会出现断续。
- 这说明当前剩余问题已经不完全是 chunk 级延迟，而更可能出在：
  - worker 最后尾音返回边界
  - 播放队列 flush 时机
  - assistant turn 结束判定

### 2. 句尾音频仍可能被截断

- 典型表现是：
  - 文本已经完整
  - 但最后几个字音频没有播出来
  - 下一轮或者后续状态变化时，尾音可能被吞掉
- 这说明：
  - 句尾补收集虽然已经做过一轮修正
  - 但 `end_of_turn / listen` 边界上的尾音处理仍未完全收敛

### 3. 自动打断手感不稳定

- 当前已经回退到较早一版的自动打断参数。
- 但打断与“完整播一句话”之间仍存在取舍：
  - 收得太紧，容易把一句话腰斩
  - 放得太松，又不够容易打断

### 4. 无输入保护与续说保护仍可能互相影响

- `unsolicited_speak_without_user_input_count`
- `assistant_continuation_grace_chunks`

这两类保护逻辑虽然已经做过收敛，但在长回答和讲故事场景下，仍可能影响完整播句。

## 下一步建议

下一轮收敛建议只盯两个问题，不再继续扩展功能：

1. 固定在 `audio-only` 下复现并收敛句尾截断问题。
2. 拆开“自动打断”和“完整播句”两条逻辑，单独做 turn 结束判定调优。

在这两个问题稳定之前，不建议继续增加新的交互模式或新的控制逻辑。
