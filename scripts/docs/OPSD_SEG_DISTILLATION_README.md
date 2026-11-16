# OPSD mask embedding 特征对齐

## 对齐什么

对齐 `[SEG]` token 最后一层 hidden state 经 `text_hidden_fcs` 投影后的 **256 维向量**，
它是实际输入 SAM 的 text prompt。Student 和 Teacher 使用 Student 采样的同一条回答，
各自取回答中第一个 `[SEG]`；Teacher 特权只影响其输入视图。

```text
z_s = P_theta(h_theta(x_student, student_rollout)[SEG])
z_t = stop_gradient(P_theta(h_theta(x_privileged, student_rollout)[SEG]))
L_feat = mean((z_s - z_t)^2)
L_total = L_forward_KL + mask_weight * (2 BCE + 0.5 Dice) + feat_weight * L_feat
```

首帧 mask 入口仍为当前参数 OPSD；音频两个入口在 2026-10-16 重训修正为独立冻结 SFT Teacher。
音频 Teacher 显式加载 checkpoint-5262 的 adapter 和 projector，forward 和投影使用 `eval + no_grad`。Student embedding 复用 mask 分支本次实际使用的向量；
不增加 LLM forward 或 Teacher SAM 解码。这里没有把 GT mask 编码成特征标签。

## 三个 4 卡入口

| 脚本 | Student 输入 | Teacher 特权 |
| --- | --- | --- |
| `scripts/opd/train_opsd_seg_feature_distill.sh` | 原始视频、音频、指代表达 | 额外 GT 类别 / 首帧 mask 特征对齐 |
| `scripts/opd/run_opsd_audio_mute50_segdistill_4gpu.sh` | 50% 随机静音音频，其余输入相同 | 原始音频 |
| `scripts/opd/run_opsd_audio_noise5db_segdistill_4gpu.sh` | 5 dB 加噪音频，其余输入相同 | 原始音频 |

```bash
bash scripts/opd/train_opsd_seg_feature_distill.sh
```

```bash
bash scripts/opd/run_opsd_audio_mute50_segdistill_4gpu.sh
```

```bash
bash scripts/opd/run_opsd_audio_noise5db_segdistill_4gpu.sh
```

默认从 3B SFT checkpoint-5262 独立开始，4 卡 DDP、每卡 batch 1、累积 1，完整训练集 1 epoch，
预计 3,509 step。学习率 `1e-5`，纯 on-policy、forward KL、temperature=1，
生成长度 512、总长度 4096；每 1,000 step 保存及验证，训练中验证 val 前 64 条。
首帧 mask 输出仍在 `outputs/train/opsd_first_mask_segdistill_current_r1/`；
音频输出为 `outputs/train/opsd_audio_{mute50,noise5db}_segdistill_frozen_r2/`。

## 对照实验与参数

默认 `AURORA_OPSD_SEG_DISTILL_WEIGHT=1.0`、`AURORA_OPSD_SEG_DISTILL_LOSS=mse`。
支持 `cosine` 损失（`1 - cos`）。`RUN_TAG`、`OUTPUT_DIR`、`MAX_STEPS`、
`NUM_TRAIN_EPOCHS`、`CUDA_VISIBLE_DEVICES` 可通过环境变量覆盖。

为了隔离特征损失的贡献，使用**同一入口、同一 Teacher 策略与训练预算**做 weight=0 对照，例如：

```bash
AURORA_OPSD_SEG_DISTILL_WEIGHT=0 \
  bash scripts/opd/train_opsd_seg_feature_distill.sh
```

音频版本可与原 `scripts/opd/run_opsd_audio_{mute50,noise5db}_4gpu.sh` 配对，默认唯一训练差别是特征项。
历史 mask 实验的 Teacher 策略可能不同，不能仅凭其旧结果归因于特征对齐。
最终在同一测试集合比较 mIoU、F-score、类别准确率和 SEG 输出率。

日志使用 `aurora/` 前缀，包含 `seg_distill_loss`、`seg_distill_weighted_loss`、
`seg_distill_mse`、`seg_distill_cos`、`seg_distill_student_norm`、`seg_distill_teacher_norm`、
`seg_distill_rows`。这些是当前日志 rank 最近 batch 的诊断量；无 SEG 时损失与 rows 为 0。
MSE 降低、cosine 增大不代表分割必然改善，需连同 embedding 范数及最终分割指标判断。

## 实现细节

核心逻辑在 `swift_plugin/aurora_opsd.py`、`swift_plugin/seg_distillation.py`。
Teacher 特征始终停止梯度；缺失 SEG 使用带零损失的 projector 图；多 SEG 取第一个。
按完整序列定位特征，避免 logits 裁剪造成 hidden/label 错位；两侧回答或样本行不一致会报错。
特征开关为 0 时跳过 Teacher 特征投影。HTTP Top-K Teacher 不提供 hidden states，不能启用此项。
既有独立本地 Teacher 也兼容，其目标使用自己的 projector 并在 offload 前完成提取。
