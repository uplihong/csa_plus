# 多节点 Stage1 实验深度复盘报告（质量 + 系统并重）

实验目录：`outputs/smoke_stage1_multinode_20260222_040048`  
报告时间：2026-02-22  
分析范围：仅基于当前目录现有工件（`train.log`、`driver.log`、TensorBoard events、3 个 checkpoint）

---

## 1. 实验配置快照

### 1.1 训练与分布式配置
- 拓扑：2 节点 x 2 GPU，`world_size=4`。
- 框架：DeepSpeed `0.18.6` + PyTorch `2.8.0+cu126`。
- 设备：NVIDIA H100 80GB HBM3。
- 精度：`bf16.enabled=true`，`fp16.enabled=false`。
- Zero：`zero_optimization.stage=0`。
- 全局 batch：`128 x 4 = 512`。
- 学习率策略：`WarmupCosineLR`，`total_num_steps=100000`，`warmup_num_steps=10000`。
- checkpoint/eval 频率：每 `500` step 执行一次（本次共 200 次）。
- 数据：LibriSpeech 16k trim manifest，训练样本 `281241`，验证样本 `5567`。

### 1.2 运行完成性与产物完整性
- 训练完整到 `step=100000` 并正常停止（`Reached max steps. Stopping.`）。
- 无中途 `OOM`、`NaN/Inf`、fatal traceback。
- 当前保留 checkpoint：`iter_500`、`iter_24500`、`iter_100000`。
- `latest` 文件指向 `iter_100000`。

---

## 2. 收敛质量

### 2.1 三个关键 step 对照（Train/Loss + Val/Loss + LR）

| checkpoint | Train/Loss | Val/Loss | LR |
|---|---:|---:|---:|
| `iter_500` | 3.59375 | 1.7840787 | 6.749594e-05 |
| `iter_24500` | 0.546875 | 0.0561095 | 9.373076e-05 |
| `iter_100000` | 0.53515625 | 0.0758915 | 1.000003e-08 |

### 2.2 全程最佳点与末步劣化
- 全程最佳验证点：`step=24500`，`Val/Loss=0.0561095`。
- 末步：`step=100000`，`Val/Loss=0.0758915`。
- 相对最优劣化：`(0.0758915 - 0.0561095) / 0.0561095 = 35.3%`。

### 2.3 结论口径（固定）
- `best_for_eval = iter_24500`
- `best_for_resume_long_train = iter_100000`
- `iter_500` 仅作早期 sanity 对照

### 2.4 质量解释
- 训练损失从 `4.875`（step 0）持续降到 `~0.53`，训练拟合仍在推进。
- 验证损失在 `step=24500` 达到最低后进入平台并轻度回升（末步高于最优 35.3%）。
- 该模式符合“后期收益变小并伴随轻度泛化回退”，因此离线评估优先使用 `iter_24500`。

---

## 3. Checkpoint 体检

### 3.1 三点元信息对齐检查

| checkpoint | step 字段 | global_steps | global_samples | lr_scheduler.last_batch_iteration | dp_world_size | mp_world_size |
|---|---:|---:|---:|---:|---:|---:|
| `iter_500` | 500 | 501 | 256512 | 500 | 4 | 1 |
| `iter_24500` | 24500 | 24501 | 12544512 | 24500 | 4 | 1 |
| `iter_100000` | 100000 | 100001 | 51200512 | 100000 | 4 | 1 |

判定：
- `global_steps = step + 1` 对齐。
- `global_samples = global_steps x 512` 对齐。
- `lr_scheduler.last_batch_iteration = step` 对齐。
- 三点的分布式并行元信息一致（`dp=4`，`mp=1`）。

### 3.2 参数与优化器状态完整性
- 三个 checkpoint 的 `module` 参数键数量一致：`209`。
- 三点参数键集合完全一致：missing `0`，extra `0`。
- 优化器状态完整（ZeRO-0 单组扁平参数）：
  - `optimizer_state_dict` 包含 `state` 与 `param_groups`
  - 每个状态含 `step`、`exp_avg`、`exp_avg_sq`
  - `fp32_groups_flat` 存在且与参数规模匹配

### 3.3 参数演化一致性（500 -> 24500 -> 100000）
- 两个相邻区间内，209/209 参数键均发生变化（不是仅 `logit_scale` 变化）。
- `contrastive_loss.logit_scale`：`2.6875 -> 4.8125 -> 5.6875`。

### 3.4 `checkpoint_exclude_frozen_parameters=true` 的影响解释
- 本次配置中，`text_encoder.freeze_model=true`，文本编码器参数被冻结。
- `speech_encoder.freeze_model=true` 且默认仅冻结 feature extractor 路径。
- 因为保存时启用了 `exclude_frozen_parameters=true`，冻结参数不会写入 checkpoint。
- 本次快照中未包含 text encoder 与 frozen feature extractor 参数，属于预期行为，不是保存异常。

---

## 4. 系统性能拆账（四段量化）

### 4.1 观测基线
- `step 0 -> step 100000` 训练循环 wall time：`46975.007s`。
- `driver start -> training stop` 端到端：`47324.686s`。
- 稳态（约 `step 5k~98k`）观测吞吐：约 `1113.7 samples/s`（`0.4597 s/iter`）。
- 末段（`step 98k~100k`）观测吞吐：约 `549.5 samples/s`（`0.9317 s/iter`）。
- `save->val_end` 周期开销：平均 `15.628s/次`，共 `200` 次。

### 4.2 四段拆账定义
- `startup`：`driver start -> step0`。
- `steady_train`：端到端扣除固定 save/val 与末段额外抖动后的剩余训练主体时间。
- `periodic_save_val`：200 次 `save->val_end` 的累计固定开销。
- `tail_instability`：`step>=98000` 区间相对稳态基线（no-eval baseline）产生的额外开销。

### 4.3 四段结果与占比（以 `driver start -> stop` 为分母）

| segment | time(s) | 占比 |
|---|---:|---:|
| startup | 333.890 | 0.71% |
| steady_train | 42865.896 | 90.58% |
| periodic_save_val | 3125.599 | 6.60% |
| tail_instability | 999.301 | 2.11% |
| **总计** | **47324.686** | **100.00%** |

### 4.4 性能结论与优先级
1. **先降固定开销**：`periodic_save_val` 占比 6.60%，且频率高（每 500 step 一次）。  
2. **再定位尾段抖动**：`tail_instability` 虽占比 2.11%，但集中发生在 `step>=98200`，对收尾吞吐影响显著。  
3. 稳态主体计算段（90.58%）相对稳定，优先级低于前两项。

---

## 5. 稳定性判定（告警分级）

### 5.1 本次观测
- 训练主过程无 P0/P1 故障（无 hang/crash/OOM）。
- 退出阶段出现以下告警：
  - `destroy_process_group() was not called before program exit`
  - worker 侧 `TCPStore recvValue failed ... got 0 bytes`（主节点先退出后的收尾告警）

### 5.2 告警分级
- **P2 可接受噪声**（本次命中）
  - 发生在训练结束后收尾阶段。
  - 不影响本次 `step=100000` 收敛结果有效性。
- **P0/P1 关键故障**（本次未见）
  - 中途 hang、rank crash、OOM、不可恢复通信失败。

### 5.3 判定结论
- 本次 run 结果有效，可用于质量/性能对比与 checkpoint 选型。
- 建议补显式分布式清理以降低退出告警噪声，便于后续自动化监控判定。

---

## 6. 行动建议（后续 A/B 固定顺序）

### 实验 1（优先）
目标：降低固定 `save+val` 开销。  
改动：`validation_every_steps` 与 `checkpoint_every_steps` 从 `500` 放宽到 `2000`（或 `5000`）。  
其余超参保持不变。  
验收：稳态吞吐不低于当前基线，端到端 wall time 明显下降。

### 实验 2
目标：验证尾段抖动是否与 checkpoint I/O 路径相关。  
改动：仅将 checkpoint 输出切到更高带宽/更低抖动存储路径。  
其余超参保持不变。  
验收：`step_p90`/`iter_p90` 在尾段显著回落。

### 实验 3
目标：清理退出噪声告警。  
改动：训练结束显式调用 `dist.destroy_process_group()`（全 rank 一致收尾）。  
其余超参保持不变。  
验收：`destroy_process_group` / `TCPStore recv` 收尾告警消失或显著减少，且无新失败。

### 实验 3 最小验证步骤
1. 启动一个最小多节点 smoke（保留与基线一致的分布式拓扑，缩短步数）。
```bash
REPO=/code/csa_plus
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/outputs/verify_pg_destroy_${TS}
mkdir -p "${OUT}"
cd "${REPO}"

deepspeed \
  --hostfile /etc/deepspeed/hostfile \
  --launcher pdsh \
  train.py \
  +experiment=limit_longest_1-3_stage1_bf16 \
  experiment_output_dir="${OUT}" \
  hydra.run.dir="${OUT}" \
  ++train.max_step_iterations=1000 \
  ++train.log_every_steps=20 \
  ++train.validation_every_steps=500 \
  ++train.checkpoint_every_steps=500 \
  ++train.checkpoint_save_latest=true \
  ++train.checkpoint_exclude_frozen_parameters=true \
  ++train.checkpoint_tag_style=iter \
  ++train.deterministic=false \
  ++train.cudnn_benchmark=false \
  ++train.enable_cuda_sync_timing=false \
  ++train.timing_rank_scope=rank0 \
  ++train.enable_eta_logging=true \
  ++train.eta_distributed_mode=rank0 \
  ++train.eta_min_samples=10 \
  ++train.data.num_workers=8 \
  ++train.data.prefetch_factor=4 \
  ++train.evaluation.eval_batch_size=64 \
  ++dataset.root_dir=data/LibriSpeech/LibriSpeech_16k_trim \
  ++dataset.manifest_path=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
  ++dataset.use_trim=false \
  ++dataset.offline_trimmed=true \
  ++train.pretrained_model_checkpoint=data/weights/csa/ckpt_epoch_8.pth \
  ++model.speech_encoder.pretrained_path=data/weights/wav2vec2-base \
  ++model.text_encoder.pretrained_path=data/weights/bert-base-uncased \
  ++model.speech_encoder.attn_implementation=eager \
  ++model.text_encoder.attn_implementation=eager \
  ++model.speech_encoder.torch_dtype=bf16 \
  ++model.text_encoder.torch_dtype=bf16 \
  deepspeed_config_yaml.zero_optimization.stage=0 \
  deepspeed_config_yaml.train_micro_batch_size_per_gpu=64 \
  deepspeed_config_yaml.wall_clock_breakdown=false \
  deepspeed_config_yaml.bf16.enabled=true \
  deepspeed_config_yaml.fp16.enabled=false \
  2>&1 | tee "${OUT}/driver.log"
```
2. 检查训练是否正常到达停止条件与进程退出。
```bash
rg -n "Reached max steps\\. Stopping|exits successfully" "${OUT}/train.log" "${OUT}/driver.log"
```
3. 检查退出噪声告警是否消失或显著减少。
```bash
rg -n "destroy_process_group\\(\\) was not called|recvValue failed|Failed to check the \"should dump\" flag on TCPStore" "${OUT}/driver.log"
```
4. 判定标准：
- 至少出现 `Reached max steps. Stopping.` 与各 rank `Process ... exits successfully`。
- 不再出现 `destroy_process_group() was not called`。
- `TCPStore recvValue failed` 与 `should dump flag` 告警应消失；若偶发残留，需对比历史 run 频次确认显著下降。

### 实验 3 验证结果（已完成）
- 实验目录：`outputs/verify_pg_destroy_20260224_074227`
- 运行版本：`fix/destroy_process_group@ff0f603`
- 分布式规模：`world_size=2`（2 节点 x 每节点 1 GPU），`micro_batch=64`，全局 batch=`128`
- 训练配置：`max_step_iterations=1000`，`validation_every_steps=500`，`checkpoint_every_steps=500`

关键结果：
- 完成状态：正常到达 `Reached max steps. Stopping.`。
- 收尾清理：出现 `Destroyed torch.distributed process group on rank=0/2`。
- 进程退出：2/2 rank 均 `exits successfully`。
- 告警计数（driver.log）：
  - `destroy_process_group() was not called`: **0**
  - `recvValue failed`: **0**
  - `Failed to check the "should dump" flag on TCPStore`: **0**
- 吞吐（step0->step1000）：
  - `sec/iter = 0.2546`
  - `iters/s = 3.9278`
  - `samples/s (global batch=256) = 1005.5`

结论：
- 实验 3 目标达成。显式 `dist.destroy_process_group()` 后，先前在退出阶段出现的 NCCL/TCPStore 收尾噪声在该验证 run 中未复现。
- 该改动不影响训练主流程完成性与 checkpoint/validation 路径，属于低风险稳定性增强。

---

## 附：验收标准（执行后对照）

1. 质量验收  
- `iter_24500` 仍为验证最优，或与最优差距仅在统计噪声范围。
- 三个 checkpoint 均可成功加载并继续推理/训练。

2. 性能验收  
- 稳态吞吐不低于当前基线。
- 放宽 save/val 频率后，端到端时间下降。

3. 稳定性验收  
- 无中途故障（hang/crash/OOM）。
- 退出阶段告警显著减少或消失。

4. 可复现性验收  
- 报告关键数字可从 `train.log`、`driver.log`、events、checkpoint 元信息交叉复算得到一致结果。

---

## 假设与默认项

1. 仅基于当前目录工件分析，不依赖外部监控系统。  
2. 目标是“选后续最有价值 checkpoint + 找吞吐瓶颈优先级”，不是复训。  
3. 退出阶段告警不作为本次 run 失败判据，但列为工程整改项。  
4. 吞吐换算口径固定为 `global batch = 512`。
