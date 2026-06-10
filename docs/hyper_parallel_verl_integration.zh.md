# HyperParallel 接入 verl 说明

本文记录这次在当前代码库中把 HyperParallel 初步接入 verl 的工作，包括适配改动、目前还缺失的能力、HyperParallel 侧可以改进的接口，以及用 GSM8K + Qwen3 0.6B 跑的一组小批量 GRPO smoke 对比结果。

## 目标

verl 现有训练后端主要走两条路径：

- FSDP/DP 路径
- Megatron 路径

这次新增了第三个后端 `hyper`，使普通 PPO/GRPO 任务可以通过 Hydra 选择 HyperParallel：

```bash
python3 -m verl.trainer.main_ppo model_engine=hyper ...
```

当前实现是一个保守的第一版适配：尽量复用 verl 已有的 HF/FSDP PPO engine 接口，只把分布式模型包装替换成 HyperParallel HSDP。它已经可以跑通小规模 RL smoke，但还不是 FSDP 和 Megatron 的完整生产级替代。

## 新增和修改的文件

新增 HyperParallel engine：

- `verl/workers/engine/hyper/__init__.py`
- `verl/workers/engine/hyper/transformer_impl.py`

新增 Hydra 配置：

- `verl/trainer/config/model_engine/hyper.yaml`
- `verl/trainer/config/engine/hyper.yaml`
- `verl/trainer/config/actor/hyper_actor.yaml`
- `verl/trainer/config/ref/hyper_ref.yaml`
- `verl/trainer/config/critic/hyper_critic.yaml`

新增验证脚本：

- `run_qwen3_0.6b_grpo_gsm8k_hyper.sh`

修改 verl 现有文件：

- `verl/workers/config/engine.py`
- `verl/workers/engine/__init__.py`
- `verl/trainer/main_ppo.py`
- `verl/workers/engine_workers.py`
- `verl/utils/tensordict_utils.py`
- `verl/utils/checkpoint/hyper_checkpoint_manager.py`

## 适配做了什么

### 1. 新增 `HyperEngineConfig`

`HyperEngineConfig` 继承自 `FSDPEngineConfig`，这样 actor/ref/critic 的 dataclass 和 PPO worker 可以继续使用现有字段，不需要大面积改 trainer 逻辑。

它固定 `strategy='hyper'`，并新增少量 HyperParallel 相关配置：

- `comm_fusion`
- `comm_fusion_zero_copy`
- `wrap_root_only`
- `apply_grad_on_fp32_main_grad`

本次 smoke 配置里使用 `model_dtype=bfloat16` 和 `dtype=bfloat16`。这是为了让 Qwen3 0.6B 在当前单张 24 GB GPU 上更稳地跑通。

### 2. 注册新的 engine backend

`verl/workers/engine/__init__.py` 中新增了对 `verl.workers.engine.hyper` 的导入，从而注册：

- `HyperEngineWithLMHead`
- `HyperEngineWithValueHead`

二者都注册在 backend `hyper` 下，支持 `cuda` 和 `npu` device。

### 3. 实现 Hyper engine

`HyperEngine` 继承 verl 的 `FSDPEngine`，只覆盖必须替换后端的部分：

- 导入路径：设置 `HYPER_PARALLEL_PLATFORM=torch`，如果环境中没有安装 `hyper_parallel`，则 fallback 到 `/root/verl_hyper/hyper-parallel`。
- 模型包装：`_build_fsdp_module()` 使用 `hyper_parallel.fully_shard()`，mesh 来自 `init_device_mesh()`。
- 混合精度：将 verl 配置里的 dtype 映射为 HyperParallel `MixedPrecisionPolicy`。
- CPU offload：把 verl 的 offload 配置映射到 HyperParallel `CPUOffloadPolicy`。
- optimizer step：用 `SkipDTensorDispatch()` 包住 `optimizer.step()`，避免 Hyper DTensor 参数进入标准 torch optimizer 路径时触发 dispatch 问题。
- grad clipping：使用 `hyper_parallel.core.utils.clip_grad_norm_()`。
- rollout 权重同步：`get_per_tensor_param()` 调用 HyperParallel 的 `get_model_state_dict()` 取 full state dict，再通过 verl 的 `convert_weight_keys()` 转换参数名，最后产出 bf16 tensor 给 vLLM 同步。

目前 `save_checkpoint()` 和 `load_checkpoint()` 还没有实现，会显式抛出 `NotImplementedError`。验证脚本里设置了 `trainer.save_freq=-1`，因此 smoke 不会走 checkpoint。

### 4. trainer worker 路由

`main_ppo.py` 中新增了 `strategy=hyper` 的路由：

- actor/rollout/ref 使用 `ActorRolloutRefWorker`
- critic 如果启用，则使用 `TrainingWorker`

这条路径和 verl 新一些的 engine worker 架构一致，类似 VeOmni/Torchtitan 的接入方式。

### 5. Hyper checkpoint manager

新增 `HyperCheckpointManager`，让 Hyper 路径不再直接在 `save_checkpoint()` / `load_checkpoint()` 抛 `NotImplementedError`。当前格式是 Hyper 专用的 lightweight checkpoint：

- `hyper_model_world_size_${world_size}_rank_${rank}.pt`：通过 HyperParallel `get_model_state_dict(..., full_state_dict=True)` 保存模型。
- `hyper_optim_world_size_${world_size}_rank_${rank}.pt`：保存 torch optimizer state dict。
- `hyper_extra_state_world_size_${world_size}_rank_${rank}.pt`：保存 lr scheduler 和 RNG。
- `huggingface/`：保存 tokenizer、generation config 和模型 config，便于后续导出或排查。
- `hyper_config.json`：记录 Hyper checkpoint 格式和 world size。

这不是伪装成 FSDP checkpoint，而是单独的 Hyper checkpoint 格式，避免不同后端的 checkpoint 被误读。保存路径已经用 Qwen3 0.6B + GSM8K + 1 step GRPO 实际验证过。

### 6. DataProto 和 TensorDict 兼容

新的 engine worker 期望进入 engine 前的数据是 `TensorDict`，但 trainer 的一些调用仍然传 `DataProto`。因此补了两处兼容：

- `tensordict_utils.pop()` 现在可以处理 `DataProto.batch` 中缺失的 key，例如 `no_lora_adapter`。
- `engine_workers.py` 在输入为 `DataProto` 时，会把默认非 tensor 字段写入 `DataProto.batch`，并把内部 `TensorDict` 传给 `engine.train_batch()` / `engine.infer_batch()`。

另外，`compute_log_prob` 路径会补齐缺失的 inference metadata：

- `compute_loss=False`
- `calculate_entropy=True`
- `temperature=1.0`
- 如果没有 `loss_mask`，则使用 `response_mask`

最终 smoke 脚本还设置了：

```bash
trainer.use_legacy_worker_impl=disable
```

这样 trainer 会先走现有的 no-padding TensorDict 转换路径，再进入 engine worker。否则 padded `DataProto` 会直接进入 FSDP/Hyper forward 路径，和 nested tensor 预期不匹配。

## 当前限制

这次接入已经能跑通功能验证，但还不是完整生产版本：

- 已新增实验性 Hyper checkpoint manager，并验证了单卡保存路径；恢复训练、多卡 world size 变化、以及更长训练的 checkpoint/load 还需要继续验证。
- optimizer state 目前用普通 torch optimizer state dict 保存；如果 Hyper 后续提供官方 optimizer checkpoint API，应替换为官方格式。
- rollout 权重同步现在通过 full state dict 实现。0.6B smoke 可以接受，但大模型 RL 每步同步权重时成本会很高。
- 当前把 Hyper HSDP 当作 FSDP 风格后端使用，还没有把 Hyper 的 TP/PP/CP 等模型并行能力暴露到 verl 的 resource pool 和 Hydra 配置中。
- verl 现有 FSDP CPU/GPU offload helper 不理解 Hyper DTensor ownership，所以 Hyper 路径没有直接复用这些 helper。
- 目前默认 `wrap_root_only=True`，这是最稳妥的选择。更细粒度的 transformer block wrapping 需要在多卡和不同模型结构上继续验证。
- 当前 smoke 为了在单张 24 GB GPU 上稳定跑通，关闭了 KL/ref 加载：`actor.use_kl_loss=False`。

## HyperParallel 目前缺少或不够顺手的接口

这次适配过程中，下面这些接口缺口让 verl 侧适配变得更“侵入式”：

- 缺少正式的 verl 兼容 checkpoint contract。当前 verl 侧已有实验性 manager，但更理想的是 Hyper 官方提供模型和 optimizer 的 save/load API，并自然对接 verl checkpoint manager 与 `torch.distributed.checkpoint` 语义。
- 缺少直接面向 rollout sync 的 streamed state-dict 或 named-parameter export API。现在 full state dict 对大模型每步同步来说成本太高。
- optimizer 集成需要手动用 `SkipDTensorDispatch()` 包住 `optimizer.step()`。如果 HyperParallel 提供 optimizer wrapper 或构造期 context manager，会更干净。
- device/offload 生命周期 hook 与 verl 的 `engine.to()`、`train_mode()`、`eval_mode()`、colocated rollout memory 管理还没有完全对齐。
- verl adapter 需要引用 `hyper_parallel.core.fully_shard.api.get_model_state_dict` 这类较内部的模块。更稳定的 public namespace 可以降低后续升级风险。
- DTensor dispatch、state dict mode、offload policy 等错误信息还可以更具体，方便训练框架定位问题。

## 对 HyperParallel 的改进建议

为了让 HyperParallel 更适合作为 verl 的一等后端，建议优先补这些能力：

- 提供稳定的 `save_model_state`、`load_model_state`、`save_optimizer_state`、`load_optimizer_state` API，并兼容 `torch.distributed.checkpoint` 的常见选项。
- 提供类似 `iter_named_parameters_for_sync()` 的 streamed 参数导出接口，支持 canonical HF 参数名、dtype 转换和 device 转换，避免每步 materialize full state dict。
- 提供标准 torch optimizer 的集成 helper，避免每个框架自己手动包 `SkipDTensorDispatch()`。
- 暴露 mesh/rank/group helper，直接提供 verl 需要的 data parallel size、rank、process group，以及可选的 model-parallel subgroup。
- 提供官方 HF transformer auto-wrap policy，最好支持 transformer-layer 粒度，而不是让接入方在 root-only 和 leaf wrapping 之间自己试。
- 增加 single-GPU/no-shard fast path。world size 1 的 smoke/debug 很常见，减少不必要的 distributed/state-dict 开销可以明显提升迭代效率。
- 提供 memory 和 communication counters，并能直接接入 verl 现有 metrics 路径。

## 单卡能不能测试并行

单卡可以测试“后端是否能接入 verl”和“world size 1 时是否能正确退化”，但不能测试真正的并行收益。原因是单卡没有跨 rank 参数切分、all-gather、reduce-scatter、跨卡 optimizer state sharding 和通信重叠压力。

因此单卡适合验证：

- 配置、worker 路由和 engine registry 是否正确。
- Hyper HSDP 包装后的 forward/backward/optimizer step 是否能跑通。
- rollout 权重同步是否能把参数交给 vLLM。
- checkpoint save/load 的基础格式是否能走通。

单卡不适合验证：

- Hyper 相比 FSDP/Megatron 的真实多卡性能优势。
- HSDP shard 后的通信效率。
- 多维 mesh、TP/PP/CP、comm fusion、zero-copy fusion 的收益。
- 大模型下 full state dict sync 的实际瓶颈。

## 小批量 GRPO 验证设置

三组实验都在同一台机器、同一张 GPU 上跑，使用 `CUDA_VISIBLE_DEVICES=5`。

公共配置：

- 模型：`/root/workspace/ft/qwen3-0.6b-cold-start-sft`
- 数据：`/root/data/gsm8k/train.parquet`、`/root/data/gsm8k/test.parquet`
- 算法：GRPO
- GPU 数：1
- `data.train_batch_size=4`
- `data.max_prompt_length=512`
- `data.max_response_length=256`
- `actor_rollout_ref.rollout.n=2`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.enforce_eager=True`
- `actor_rollout_ref.actor.use_kl_loss=False`
- `algorithm.use_kl_in_reward=False`
- `trainer.total_training_steps=1`
- `trainer.save_freq=-1`
- `trainer.test_freq=-1`

注意：这只是 one-step smoke benchmark。生成是随机的，而且不同后端采样到的 response token 数不同，所以 `perf/throughput` 和 `time_per_step` 只能作为初步信号，不能当作严格性能结论。

## 额外 checkpoint 验证

新增 checkpoint manager 后，又跑了一次 Hyper 1 step GRPO，并设置：

```bash
trainer.save_freq=1 trainer.total_training_steps=1 trainer.test_freq=-1
```

保存成功，actor checkpoint 目录为：

```text
/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k_hyper/global_step_1/actor
```

关键文件：

- `hyper_model_world_size_1_rank_0.pt`，约 1.5 GB
- `hyper_optim_world_size_1_rank_0.pt`，约 2.3 GB
- `hyper_extra_state_world_size_1_rank_0.pt`
- `hyper_config.json`
- `huggingface/` tokenizer/config 文件

该次保存耗时约 6.66 秒，包含 full state dict gather/offload 和 optimizer state 写盘。

## 结果

| 后端 | Engine 路径 | Dtype | Step time | Throughput | Total tokens | Gen | Old log prob | Actor update | Weight sync |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HyperParallel | `model_engine=hyper` | bf16 | 19.49 s | 77.64 tok/s | 1513 | 13.25 s | 2.39 s | 2.79 s | 1.04 s |
| FSDP | `model_engine=dp`, `strategy=fsdp` | bf16 | 19.33 s | 72.21 tok/s | 1396 | 12.64 s | 2.73 s | 2.87 s | 1.08 s |
| FSDP | `model_engine=dp`, `strategy=fsdp` | fp32 model init | 20.54 s | 78.97 tok/s | 1622 | 13.05 s | 2.87 s | 3.20 s | 1.39 s |
| Megatron | `ppo_megatron_trainer.yaml` | bf16 | 19.28 s | 69.54 tok/s | 1341 | 11.81 s | 3.99 s | 2.06 s | 1.40 s |

相对 bf16 FSDP，这次 Hyper 的端到端 step time 基本持平，慢约 0.8%；token throughput 高约 7.5%，但这个数字会受到 response token 数不同的影响。模型侧 timing 中，Hyper 的 old-log-prob 更快，actor update 略快，weight sync 接近。

相对单 GPU Megatron，这次 Hyper 的 step time 基本持平，慢约 1.1%；sample 内 token throughput 更高；old-log-prob 和 weight sync 更快，但 actor update 慢于 Megatron。

## 结论

在这个单 GPU、0.6B、1 step 的 smoke 工作负载上，HyperParallel 已经可以作为 verl 后端跑通 GRPO，并且性能和 FSDP/Megatron 在同一档。

不过，这组实验还不能证明 HyperParallel 官网中宣称的大规模性能优势。当前 world size 为 1，没有真实多卡 sharding 和通信压力。更有意义的下一步是做多 GPU benchmark：固定 prompt/sample 设置，增大 batch，多轮重复统计，并单独观察 GPU memory、rollout weight sync、old-log-prob、actor update 等指标。

## 验证命令

Hyper smoke：

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=disabled \
  bash run_qwen3_0.6b_grpo_gsm8k_hyper.sh \
  actor_rollout_ref.rollout.enforce_eager=True
```

FSDP bf16 smoke 使用默认 DP/FSDP 配置，并额外覆盖：

```bash
model_engine=dp \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
trainer.use_legacy_worker_impl=disable \
trainer.total_training_steps=1
```

Megatron smoke 使用：

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_megatron_trainer.yaml \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps=1
```
