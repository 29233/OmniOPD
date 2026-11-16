# ms-swift：音频 / 类别双特权视图 OPSD (Dual Audio/Class Gated OPSD)

## 1. 媒体与帧率协议

train/val/test_s/test_u 共 3566 个不同片段：每段 clip ID 对应 10 秒，磁盘均有 10 张帧图，
音频约 10 秒；名义帧密度是 **1 FPS**。
对于曾使用 `FPS=2.0`、`AURORA_VIDEO_FPS=2.0` 训练的旧版 3B 模型 checkpoint，评测时若使用数据名义
1 FPS，需在评测计划清单中显式记录历史训练时间轴与评测时间轴的差异。

## 2. 训练入口

4 卡、1 epoch 训练入口：

```bash
bash scripts/opd/run_opsd_audio_class_gate_4gpu.sh
```

8 卡、1 epoch 训练入口（Forward KL）：

```bash
bash scripts/opd/run_opsd_audio_class_gate_8gpu.sh
```

8 卡、1 epoch 消融入口（Reverse KL）：

```bash
bash scripts/opd/run_opsd_audio_class_gate_reverse_8gpu.sh
```

8 卡入口启动 8 个 DDP Student 副本；每张卡额外加载同一份冻结 SFT Teacher，并用它计算类别和原始音频两个特权视图。Teacher 默认 CPU offload 以控制显存占用。

Reverse KL 消融版本使用相同 mute50 数据、Teacher 视图、Student rollout、gate 和参数设置，仅将分布损失从 Forward KL 改为完整词表 Reverse KL：

```text
KL_reverse[k,t] = KL(p_student[t] || q_teacher[k,t])
```

## 3. 模型和输入视图

Student 与 Teacher 从同一 SFT checkpoint 初始化。Teacher 是单独加载的 SFT 模型与 adapter，参数强制冻结；每步通过同一个 Teacher 分别计算类别和原始音频视图。每个 batch 都重新从当前 Student 采样一次回答；三个视图使用同一回答 token IDs。

| 视图 | 视频 / 指代表达 | 音频 | 类别提示 |
|---|---|---|---|
| Student / 无特权对照 | 原始 | 随机 mute 总计 50%（100 ms 区块） | 无 |
| 类别 Teacher | 同 Student | 同 Student 的 mute50 音频 | GT 类别 |
| 音频 Teacher | 同 Student | 原始音频 | 无 |

GT mask 只用于 Student 分割损失，不进入 Teacher 输入。类别 Teacher 的提示是：

```text
(The target object is: {class}. After understanding this, solve with your own approach.)
```

`null` / `unknown` 无类别提示，且类别 Teacher 在 loss/gate 中无效；音频 Teacher 仍有效。如果原始/退化音频编码结果完全相同（例如静音），自动关闭该样本的音频 Teacher。

## 4. 门控和损失

令 `p` 为普通 Student 分布，`qC` / `qA` 为冻结 Teacher 的特权分布。所有分布在相同 Student 回答前缀处计算，沿用同一词表和末尾 EOS 监督；`[SEG]` 参与语言蒸馏，并使用真实 `[SEG]` hidden state 做分割监督。

```text
delta[k,t] = log q[k,t](sampled_token[t]) - log stopgrad(p[t](sampled_token[t]))
gate[k,t]  = valid[k] AND delta[k,t] > threshold[k]
a[k,t]     = valid[k] * clip((delta[k,t] - threshold[k])_+ / scale[k], 0, cap)
route[k,t] = a[k,t] / sum_k a[k,t]            # 全部关闭时 route=0
KL[k,t]    = KL(q[k,t] || p[t])              # 全词表 Forward KL（或 Reverse KL），直接反传
base_loss  = sum(valid * KL) / max(1, sum(valid))
gate_loss  = sum(a * KL) / max(eps, sum(a))   # 空 gate 精确为 0
KD         = base_weight * base_loss + gate_weight * gate_loss
total      = KD + 2 * BCE + 0.5 * Dice
```

默认 `thresholdC=thresholdA=0.02`、`scaleC=scaleA=0.1`、`cap=2`、`base_weight=gate_weight=1`、`temperature=1`。`AURORA_DUAL_GATE_WEIGHT=0` 可关闭额外 gate 分支，用于基础双视图蒸馏对照。Gate、增益、Teacher 分布全部停止梯度。
