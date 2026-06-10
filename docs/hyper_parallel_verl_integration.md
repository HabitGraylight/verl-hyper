# HyperParallel adapter for verl

This document records the first HyperParallel integration in this checkout, the compatibility work that was needed in verl, gaps that remain in the HyperParallel-facing API surface, and a small GRPO smoke benchmark against FSDP and Megatron.

## Goal

The original verl training stack mainly uses the FSDP/DP path and the Megatron path. The integration adds a third backend named `hyper` so a normal PPO/GRPO job can select HyperParallel through Hydra:

```bash
python3 -m verl.trainer.main_ppo model_engine=hyper ...
```

The current implementation is deliberately conservative: it reuses verl's existing HF/FSDP PPO engine surface and swaps the distributed model wrapper to HyperParallel HSDP. It is suitable for smoke tests and incremental optimization, but it is not yet a full replacement for all FSDP and Megatron features.

## Files added or changed

Added HyperParallel engine files:

- `verl/workers/engine/hyper/__init__.py`
- `verl/workers/engine/hyper/transformer_impl.py`

Added Hydra config entries:

- `verl/trainer/config/model_engine/hyper.yaml`
- `verl/trainer/config/engine/hyper.yaml`
- `verl/trainer/config/actor/hyper_actor.yaml`
- `verl/trainer/config/ref/hyper_ref.yaml`
- `verl/trainer/config/critic/hyper_critic.yaml`

Added smoke script:

- `run_qwen3_0.6b_grpo_gsm8k_hyper.sh`

Changed existing verl files:

- `verl/workers/config/engine.py`
- `verl/workers/engine/__init__.py`
- `verl/trainer/main_ppo.py`
- `verl/workers/engine_workers.py`
- `verl/utils/tensordict_utils.py`
- `verl/utils/checkpoint/hyper_checkpoint_manager.py`

## What changed for the adapter

### 1. New `HyperEngineConfig`

`HyperEngineConfig` subclasses `FSDPEngineConfig` so the existing actor/ref/critic dataclasses and PPO worker code can keep using the same field names. It fixes `strategy='hyper'` and adds a small set of Hyper-specific options:

- `comm_fusion`
- `comm_fusion_zero_copy`
- `wrap_root_only`
- `apply_grad_on_fp32_main_grad`

The smoke config sets `model_dtype=bfloat16` and `dtype=bfloat16`. This was necessary on the available single 24 GB GPU; fp32 actor construction was not useful for the target smoke run.

### 2. Engine registry wiring

`verl/workers/engine/__init__.py` imports `verl.workers.engine.hyper`, which registers:

- `HyperEngineWithLMHead` for language model training
- `HyperEngineWithValueHead` for value model training

Both are registered under backend `hyper` for `cuda` and `npu` devices.

### 3. Hyper engine implementation

`HyperEngine` subclasses verl's `FSDPEngine` and overrides only the parts that need a different backend:

- Import path setup: sets `HYPER_PARALLEL_PLATFORM=torch` and falls back to the sibling checkout at `/root/verl_hyper/hyper-parallel` if `hyper_parallel` is not installed.
- Model wrapping: `_build_fsdp_module()` calls `hyper_parallel.fully_shard()` with a one-dimensional DP mesh from `init_device_mesh()`.
- Mixed precision: maps verl dtype config to HyperParallel `MixedPrecisionPolicy`.
- Offload: maps forward-only/offload policy to HyperParallel `CPUOffloadPolicy` where possible.
- Optimizer stepping: wraps `optimizer.step()` inside `SkipDTensorDispatch()` because Hyper DTensor params otherwise hit dispatch issues in the standard torch optimizer path.
- Grad clipping: uses `hyper_parallel.core.utils.clip_grad_norm_()`.
- Rollout weight sync: implements `get_per_tensor_param()` by calling `hyper_parallel.core.fully_shard.api.get_model_state_dict()` with a full state dict, converting keys through verl's `convert_weight_keys()`, and yielding bf16 tensors to vLLM sync.

Checkpoint save/load now use a lightweight Hyper-specific checkpoint manager. It writes Hyper full model state, torch optimizer state, lr scheduler/RNG state, and HF tokenizer/config metadata. Single-GPU save has been validated; resume and multi-GPU checkpoint compatibility still need more testing.

### 4. Worker selection

`main_ppo.py` now routes `strategy=hyper` through the new engine worker path:

- actor/rollout/ref uses `ActorRolloutRefWorker`
- critic, if enabled, uses `TrainingWorker`

This matches the existing engine-worker architecture used by newer backends such as VeOmni/Torchtitan.

### 5. DataProto and TensorDict compatibility

The new engine worker path expects a `TensorDict` when entering the engine, while several trainer calls still pass a `DataProto` wrapper. Two compatibility fixes were needed:

- `tensordict_utils.pop()` now handles missing keys on `DataProto.batch`, which fixed missing optional metadata such as `no_lora_adapter`.
- `engine_workers.py` assigns default non-tensor fields into `DataProto.batch` when the input is a `DataProto`, and passes that inner `TensorDict` into `engine.train_batch()` / `engine.infer_batch()`.

For `compute_log_prob`, the worker now fills missing inference metadata:

- `compute_loss=False`
- `calculate_entropy=True`
- `temperature=1.0`
- `loss_mask=response_mask` when `loss_mask` is absent

The final smoke script also sets `trainer.use_legacy_worker_impl=disable`. That forces the trainer to use the existing no-padding TensorDict conversion path before calling the engine worker. Without this, padded `DataProto` objects reached the FSDP/Hyper forward path and failed on nested tensor expectations.

## Current limitations

This is a functional first adapter, not yet a complete production engine.

- A lightweight Hyper checkpoint manager is now wired and single-GPU save is validated, but resume, long-running jobs, and multi-GPU checkpoint compatibility still need validation.
- Optimizer state is saved through the regular torch optimizer state dict. This should move to an official Hyper optimizer checkpoint API when one is available.
- Weight sync currently materializes a full state dict via `get_model_state_dict()`. That is acceptable for a 0.6B smoke test, but should be replaced by a streamed or bucketed per-tensor API for larger models.
- The adapter implements Hyper HSDP as an FSDP-style drop-in. It does not yet expose Hyper TP/PP/CP-style model parallelism through verl resource pools and Hydra configs.
- The adapter bypasses normal verl FSDP CPU/GPU offload helpers because they do not understand Hyper DTensor ownership.
- `wrap_root_only=True` is the safe default. More granular wrapping exists in the code path, but it needs validation on transformer block boundaries and multi-GPU runs.
- The current smoke run disables KL/ref loading (`actor.use_kl_loss=False`) to fit comfortably on a single 24 GB GPU.

## HyperParallel interface gaps noticed during integration

The following missing or awkward interfaces made the adapter more invasive than ideal:

- No official verl-compatible checkpoint contract: this checkout has an experimental manager, but Hyper should provide model and optimizer save/load APIs that map cleanly to verl's checkpoint manager and `torch.distributed.checkpoint` semantics.
- No direct streamed state-dict or named-parameter export API for rollout sync. A full state dict is too expensive for larger RL jobs where vLLM weights are updated every step.
- Optimizer integration requires manually wrapping `optimizer.step()` in `SkipDTensorDispatch()`. A backend-provided optimizer wrapper or context manager at construction time would be cleaner.
- Device/offload lifecycle hooks are not aligned with verl's `engine.to()`, `train_mode()`, `eval_mode()`, and colocated rollout memory choreography.
- The adapter has to know internal Hyper modules such as `hyper_parallel.core.fully_shard.api.get_model_state_dict`. A stable public namespace would reduce integration risk.
- Error messages around DTensor dispatch, unsupported state dict modes, and offload policy are not always specific enough for a training framework integration.

## Suggested HyperParallel improvements

High-value improvements for making HyperParallel a first-class verl backend:

- Provide a stable `save_model_state`, `load_model_state`, `save_optimizer_state`, and `load_optimizer_state` API compatible with `torch.distributed.checkpoint` options.
- Provide a streamed `iter_named_parameters_for_sync()` style API that yields canonical HF parameter names and supports dtype/device conversion without materializing the full model state on every rollout sync.
- Provide a documented optimizer integration helper so standard torch optimizers can step Hyper DTensor parameters without manual monkey-patching.
- Expose mesh/rank/group helpers with the same concepts verl needs: data parallel size, rank, process group, and optional model-parallel subgroups.
- Add an official HF transformer auto-wrap policy, preferably at transformer-layer granularity, instead of requiring each integration to choose root-only versus leaf-level wrapping.
- Add a single-GPU/no-shard fast path. Smoke tests and debugging often run world size 1, and avoiding unnecessary distributed/state-dict overhead makes iteration much easier.
- Publish memory and communication counters that can be logged through verl's existing metrics path.

## Can one GPU test parallelism?

One GPU can test backend wiring and the world-size-1 fallback path, but it cannot measure real parallel speedups. There is no cross-rank sharding pressure, all-gather/reduce-scatter traffic, optimizer-state partitioning, or communication overlap to optimize.

Single-GPU smoke is useful for checking config routing, forward/backward/optimizer step, rollout weight sync, and checkpoint format. It is not useful for proving Hyper-vs-FSDP/Megatron scaling advantages, TP/PP/CP behavior, comm fusion, or large-model state-dict sync cost.

## Additional checkpoint validation

After adding `HyperCheckpointManager`, I reran Hyper GRPO with `trainer.save_freq=1`. The run completed and wrote:

- `/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k_hyper/global_step_1/actor/hyper_model_world_size_1_rank_0.pt` (about 1.5 GB)
- `/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k_hyper/global_step_1/actor/hyper_optim_world_size_1_rank_0.pt` (about 2.3 GB)
- `/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k_hyper/global_step_1/actor/hyper_extra_state_world_size_1_rank_0.pt`
- `hyper_config.json` and `huggingface/` tokenizer/config metadata

The checkpoint save portion took about 6.66 seconds in that smoke run.

## Smoke benchmark setup

All runs below used the same machine and same visible GPU (`CUDA_VISIBLE_DEVICES=5`) with:

- Model: `/root/workspace/ft/qwen3-0.6b-cold-start-sft`
- Data: `/root/data/gsm8k/train.parquet`, `/root/data/gsm8k/test.parquet`
- Algorithm: GRPO
- GPUs: 1
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

Important caveat: this is a one-step smoke benchmark. Generation is stochastic and the sampled response lengths differed across runs, so `perf/throughput` and `time_per_step` are useful sanity signals but not a statistically stable performance conclusion.

## Benchmark results

| Backend | Engine path | Dtype | Step time | Throughput | Total tokens | Gen | Old log prob | Actor update | Weight sync |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HyperParallel | `model_engine=hyper` | bf16 | 19.49 s | 77.64 tok/s | 1513 | 13.25 s | 2.39 s | 2.79 s | 1.04 s |
| FSDP | `model_engine=dp`, `strategy=fsdp` | bf16 | 19.33 s | 72.21 tok/s | 1396 | 12.64 s | 2.73 s | 2.87 s | 1.08 s |
| FSDP | `model_engine=dp`, `strategy=fsdp` | fp32 model init | 20.54 s | 78.97 tok/s | 1622 | 13.05 s | 2.87 s | 3.20 s | 1.39 s |
| Megatron | `ppo_megatron_trainer.yaml` | bf16 | 19.28 s | 69.54 tok/s | 1341 | 11.81 s | 3.99 s | 2.06 s | 1.40 s |

Relative to the bf16 FSDP run, Hyper was similar on end-to-end step time (+0.8% slower by wall time) and faster on token throughput (+7.5%), but the throughput difference is affected by different sampled token counts. On model-side timings, Hyper's old-log-prob pass was faster than FSDP in this run (2.39 s vs 2.73 s), actor update was slightly faster (2.79 s vs 2.87 s), and weight sync was close (1.04 s vs 1.08 s).

Relative to the Megatron single-GPU run, Hyper had similar wall time (+1.1% slower), higher token throughput in this sample (+11.6%), faster old-log-prob computation (2.39 s vs 3.99 s), slower actor update (2.79 s vs 2.06 s), and faster weight sync (1.04 s vs 1.40 s).

## Interpretation

On this single-GPU 0.6B smoke workload, HyperParallel is already functional and roughly in the same performance band as FSDP and Megatron. This run does not demonstrate the advertised large-scale HyperParallel advantage because there is no real multi-GPU sharding pressure at world size 1. The more meaningful next benchmark is a multi-GPU run with larger batch sizes and a fixed prompt/sample set, repeated several times, while measuring GPU memory, rollout sync cost, and actor update separately.

## Commands used for validation

Hyper smoke:

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=disabled   bash run_qwen3_0.6b_grpo_gsm8k_hyper.sh   actor_rollout_ref.rollout.enforce_eager=True
```

FSDP bf16 smoke used the default DP/FSDP config with additional overrides equivalent to the Hyper smoke, especially:

```bash
model_engine=dp actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 trainer.use_legacy_worker_impl=disable trainer.total_training_steps=1
```

Megatron smoke used:

```bash
python3 -m verl.trainer.main_ppo   --config-path=config   --config-name=ppo_megatron_trainer.yaml   algorithm.adv_estimator=grpo   actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1   actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1   actor_rollout_ref.actor.megatron.context_parallel_size=1   trainer.n_gpus_per_node=1   trainer.total_training_steps=1
```
