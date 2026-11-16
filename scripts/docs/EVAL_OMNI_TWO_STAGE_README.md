# Omni 两阶段 REF-AVS 评测

统一入口：`scripts/eval/eval_omni_two_stage.py` 与 `scripts/eval/eval_omni_two_stage.sh`。
支持 `--family qwen3` 和 `--family qwen2_5`，复用评测脚本的模型及 adapter 加载方式。
注意：对于 Qwen2.5-Omni + SAM2 视频记忆模型的官方评测，推荐使用 `scripts/eval/test_qwen2_5_omni_refavs.py` 的完整回放分割路径。

## 流程

1. **generate：LLM 推理。** 批量生成完整回答。在解码实际读入 `[SEG]` 的那一步，捕获最后一层 hidden state，经训练后的 `text_hidden_fcs` 投影为 `[256]` embedding，保存到 CPU 文件。同时导出训练后的 SAM mask decoder。
2. **segment：独立分割。** 只加载 SAM、导出的 mask decoder 和 embedding，不加载 LLM、不重新编码音视频、不 teacher force 回答。按视频缓存 SAM image embedding，再逐个处理文本 prompt。
3. **summarize：分片汇总。** 按样本数加权汇总所有 shard，校验 checkpoint、数据集和生成设置相同。

保存的是已经投影后的 prompt embedding，不是 token ID；单独的 `[SEG]` ID 无法恢复其上下文特征。只在需要的一次 forward 返回 hidden states，避免保留整段生成的所有层特征。

当前批量优化作用于第一阶段；第二阶段按样本运行 mask decoder、复用图像特征。第一阶段仍通过现有 AURORA 插件加载 SAM 模块，但不会执行分割；第二阶段完全独立于语言模型。

## 环境

两种模型使用各自已经可用的环境，无需安装或升级包：

| 模型 | Python | PYTHONPATH | 推理精度 | 默认视频时间轴 |
| --- | --- | --- | --- | --- |
| Qwen3-Omni 30B | `python` | 仓库 `third_party/ms-swift` + 仓库根目录 | BF16 | 1 FPS |
| Qwen2.5-Omni 7B/3B | `python` | 仓库根目录 | FP16 | 2 FPS |

Qwen3 实测为 Transformers 4.57.6、Swift 4.5.2；Qwen2.5 为 Transformers 5.16.1、Swift 4.5.2。不要把这次 30B checkpoint 直接切换到 Qwen2.5 的环境；新版 MoE 权重结构与旧 adapter 不一致。

视频时间轴必须与训练一致。若 Qwen2.5 checkpoint 按 1 FPS 训练，传 `--video-fps 1`。已有的 `FPS` / `AURORA_VIDEO_FPS` 环境变量优先于默认值和该参数，请检查或清除旧环境变量；实际值会写入 manifest。SAM 默认处理每个样本 10 帧，使用项目的 **224×224** 配置。

## 一键执行两阶段：4 卡启动器

`scripts/eval/eval_omni_two_stage.sh` 会先等待所有生成分片结束，再启动分割、汇总。每个分片保存独立目录和日志。

Qwen3：默认两个模型副本，每个副本使用两张卡。

```bash
FAMILY=qwen3 \
CHECKPOINT=outputs/train/qwen3_omni_30b_joint_sft_4gpu/v0-20261008-154045/checkpoint-8000 \
OUTPUT_DIR=outputs/eval/two_stage/qwen3_checkpoint8000_full \
GPU_GROUPS='0,1;2,3' BATCH_SIZE=2 MAX_MEMORY_PER_GPU=60GiB \
bash scripts/eval/eval_omni_two_stage.sh
```

Qwen2.5：默认四个模型副本，每卡一个。以下示例评测 1 epoch 分割后训练；也可替换成 OPD checkpoint。

```bash
FAMILY=qwen2_5 \
CHECKPOINT=outputs/train/qwen2_5_omni_7b_seg_only_1epoch_4gpu/v0-20261010-173359/checkpoint-1755 \
OUTPUT_DIR=outputs/eval/two_stage/qwen2_5_seg_only_1epoch_full \
GPU_GROUPS='0;1;2;3' BATCH_SIZE=2 \
bash scripts/eval/eval_omni_two_stage.sh
```

常用环境变量：

| 变量 | 默认 / 用途 |
| --- | --- |
| `STAGE` | `all`；可设 `generate` 或 `segment` 单独运行 |
| `GPU_GROUPS` | 分号分隔副本，逗号分隔副本内 GPU；各组应互不重叠 |
| `BATCH_SIZE` | `1`；确认显存容量后可增大 |
| `MAX_SAMPLES` | `-1`，完整评测；正数表示每个 split 的前 N 个样本，再按 shard 划分 |
| `SPLITS` | `test_s test_u` |
| `MAX_NEW_TOKENS` | Qwen3=`256`，Qwen2.5=`512`，与原四卡启动脚本一致 |
| `VIDEO_FPS` | 覆盖模型家族默认值；已有 FPS 环境变量仍优先 |
| `CACHE_SIZE` | `32` 个视频，缓存占用 GPU 显存；`0` 关闭 |
| `SAVE_MASKS` | `0`；设 `1` 保存二值 PNG |
| `RESUME` | `0`；设 `1` 续跑，须保持原生成配置 |
| `RETRY_ERRORS` | `0`；配合 `RESUME=1` 重试失败记录 |
| `DRY_RUN` | `0`；设 `1` 打印命令，不加载模型、不验证硬件 |
| `MODEL_PATH` / `DATA_DIR` / `SAM_CHECKPOINT` / `PYTHON_BIN` | 覆盖路径 |

只用两张卡时，Qwen3 设 `GPU_GROUPS='0,1'`；Qwen2.5 设 `GPU_GROUPS='0;1'`。启动器不使用 torchrun：每个进程就是一个独立推理副本。分割时每个副本只使用其 GPU 组的第一张卡。

## 分别运行两个阶段

### Qwen3 生成

```bash
PYTHONPATH=third_party/ms-swift:. \
CUDA_VISIBLE_DEVICES=0,1 python \
scripts/eval/eval_omni_two_stage.py generate \
  --family qwen3 \
  --model models/Qwen3-Omni-30B-A3B-Instruct \
  --checkpoint outputs/train/qwen3_omni_30b_joint_sft_4gpu/v0-20261008-154045/checkpoint-8000 \
  --data-dir outputs/data/refavs \
  --output outputs/eval/two_stage/qwen3_checkpoint8000 \
  --splits test_s test_u --batch-size 2 --max-memory-per-gpu 60GiB
```

### Qwen2.5 生成

```bash
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 python \
scripts/eval/eval_omni_two_stage.py generate \
  --family qwen2_5 \
  --model models/Qwen2___5-Omni-7B \
  --checkpoint outputs/train/qwen2_5_omni_7b_opsd_class_4gpu_split_teacher/v0-20261010-071623/checkpoint-1000 \
  --data-dir outputs/data/refavs \
  --output outputs/eval/two_stage/qwen2_5_checkpoint1000 \
  --splits test_s test_u --batch-size 2
```

### 独立分割：两种家族共用

将 `--output` 指向刚才生成的目录。无需传 `--family`、`--model` 或原始 adapter checkpoint，使用已导出的 decoder。SAM 模块精度从生成 manifest 恢复，保持与第一阶段加载的 SAM 一致。

```bash
CUDA_VISIBLE_DEVICES=0 python \
scripts/eval/eval_omni_two_stage.py segment \
  --data-dir outputs/data/refavs \
  --output outputs/eval/two_stage/qwen3_checkpoint8000 \
  --sam-checkpoint models/sam_vit_h_4b8939.pth \
  --device cuda:0 --cache-size 32 --save-masks
```

初次 smoke 可在 generate 中加 `--splits test_s --max-samples 2`。正式全量评测使用新的目录并删除 `--max-samples`，否则 manifest 校验会拒绝把抽样与全量混在一起。

手动分片时，分别传 `--shard 0 --shards 2`、`--shard 1 --shards 2`，且使用不同的 output。每个分片独立执行 segment 后：

```bash
python scripts/eval/eval_omni_two_stage.py summarize \
  --inputs /path/to/run/shard-0 /path/to/run/shard-1 --output /path/to/run
```

## 产物与续跑

```text
output/
  generation_manifest.json             # checkpoint 身份、数据 SHA256、配置、软件版本、SAM 精度
  mask_decoder.safetensors             # 训练后的 decoder，带 SHA256 校验
  embeddings/test_s/00000000_....pt      # FP32 CPU [256] 投影结果
  test_s.jsonl                         # 完整回答、原始生成 token IDs、状态和 embedding 相对路径
  summary_test_s.generated.json
  segmentation_manifest.json
  test_s.segmented.jsonl               # IoU/Dice、缓存命中、耗时、生成状态
  summary_test_s.segmented.json
  masks/test_s/00000000/00.png          # 可选
```

`test_u` 的结构相同。embedding 路径相对于 output，整个目录搬移后仍能读取。SAM 基础权重身份校验使用绝对路径、文件大小和修改时间；adapter 身份也使用这三项，并额外校验 adapter config SHA256。导出的 decoder 和数据集使用内容 SHA256。

- 同一命令增加 `--resume` 即可续跑；配置、checkpoint 或数据集不同会拒绝混用。
- 每条结果立即 flush/fsync，embedding 先写临时文件再原子替换。恢复时只修复被截断的末行；完整但损坏的 JSON 行会报错。
- 默认跳过已记录的状态；`--resume --retry-errors` 重试 `error` / `capture_missing`，第二阶段重试 `error`。重试采用追加日志，同一 `case_index` 的最后一条为准。
- 新生成结果对应的分割需继续执行 `segment --resume`。若重试前某条生成结果已被分割为 skipped/error，新的生成结果会使该分割记录失效并重新计算。
- 更换 batch size、最大 token 数等配置请创建新目录；例如 OOM 后用较小 batch size 开新运行。
- 不要直接拼接分片 JSONL；用 `summarize` 加权汇总，避免路径失效和重复样本计分。不要让多个进程同时写同一个 output。

## 状态与指标

生成状态为 `ok`、`no_seg`、`multiple_seg`、`capture_missing`、`error`。若 `[SEG]` 恰好是停止前的最后一个 token（例如达到生成长度上限），没有后续 forward 能读入它，标记 `capture_missing`，不使用预测 `[SEG]` 时的前一个 hidden state。可增大 `--max-new-tokens` 后在新目录重跑。

分割保留 `generation_status`，状态为 `ok`、`skipped` 或 `error`；失败与缺失按 0 分计入 `mean_iou_all` / `mean_dice_all`。`*_scored` 只统计成功分割样本，必须连同 `scored`、状态计数和 `complete` 一起阅读。

- 二值阈值：`sigmoid(logits) >= 0.5`。
- 每帧计算 IoU/Dice，再按帧平均得到样本分数；最后按样本平均，与现有脚本一致。预测与 GT 都空时为 1。
- `complete` 表示所有选定样本已有生成及分割记录，包含失败记录；不代表错误数为 0。
- 按原始 `case_index` 关联数据，并核对 sample ID，避免重复 ID 导致错配。
- 缓存以精确 `sam_frame_paths` 为 key；GT 按当前样本读取，避免同一视频不同 reference 的掩码串用。
- `generation_sec` 是批次耗时总和，含媒体预处理、推理、投影；模型加载另记。`segmentation_sec` 含图像处理、decoder、指标及可选 PNG 写入。并行分片耗时相加不是任务的墙钟耗时。
