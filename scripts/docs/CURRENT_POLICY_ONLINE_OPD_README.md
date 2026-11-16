# Student-only 初筛 + 当前策略在线特权 Teacher (Current-Policy Online OPD)

## 1. Student-only 初始难度分层

使用 `scripts/data/prepare_student_only_opd_data.py` 对多组随机采样 Student 探测轨迹进行统计，不依赖离线缓存的 Teacher 探测结果：

- `rl_active`：Student 组内奖励具有非零方差；
- `hard`：Student 组内奖励持续偏低；
- `intermediate`：中等奖励水平样本；
- `mastered`：Student 组内奖励稳定偏高。

在每种模态（Audio / Audio-Visual / Visual / Semantic）内部按设定配比（如 55% / 30% / 10% / 5%）及类别词频平方根反比进行带放回采样，生成训练清单 `train.jsonl` 与完整注释 `annotated.jsonl`。

## 2. 在线当前策略特权 Teacher (`swift_plugin/aurora_online_verified_opd.py`)

1. 当前 Student 生成普通输入轨迹并计算类别 / mask reward；
2. 使用**同一个当前 policy（保留启用的 q/v LoRA adapter）**，在特权 prompt 下实时生成一条 Audio、一条 Visual Teacher 轨迹（temperature 1.0、top-p 0.9）；
3. 用相应特权 prompt 回放 Teacher response，通过训练 checkpoint 的冻结 projector / SAM2 decoder 计算 10 帧 IoU；
4. 根据当前 Student 和当前 Teacher 的类别正确性与 IoU 决定准入许可（License）与路由权重（Routing）；
5. 同一当前 policy 在特权 prompt 下对当前 Student 已采样 token 打分；
6. 原生有符号对数比 `logp_teacher - logp_student` 经阶段掩码、低似然局部筛选与准入路由后，叠加至策略优势函数；Teacher 分支全程 `no_grad`。

Visual / Semantic 模态自动禁用 Audio Teacher。

## 3. 三组受控实验对照与启动

通过 `scripts/opd/run_current_policy_opd_8gpu.sh` 的 `ARM` 环境变量切换三组严格匹配的实验分支：

| 实验 (`ARM`) | Task Advantage | Online Privileged OPD Advantage |
|---|---:|---:|
| `grpo` | 1 | 0 |
| `opd` | 0 | 1 |
| `joint` | 1 | 1 |

启动命令：

```bash
ARM=joint \
SFT_CHECKPOINT=/path/to/sam2_sft_checkpoint \
DATA_ROOT=/path/to/student_only_opd_data \
OUTPUT_DIR=/path/to/output_dir \
bash scripts/opd/run_current_policy_opd_8gpu.sh
```
