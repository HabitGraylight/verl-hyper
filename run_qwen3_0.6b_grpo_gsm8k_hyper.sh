#!/bin/bash
set -x

# HyperParallel must use its PyTorch platform path.
export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export PYTHONPATH="/root/verl_hyper/hyper-parallel:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}

MODEL_PATH=${MODEL_PATH:-"/root/workspace/ft/qwen3-0.6b-cold-start-sft"}
TRAIN_DATA=${TRAIN_DATA:-"/root/data/gsm8k/train.parquet"}
TEST_DATA=${TEST_DATA:-"/root/data/gsm8k/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k_hyper"}

mkdir -p "${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
    model_engine=hyper \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TEST_DATA}" \
    data.train_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.wrap_root_only=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_qwen3_0.6b_gsm8k_hyper' \
    trainer.experiment_name='qwen3_0.6b_hyper_smoke' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.use_legacy_worker_impl=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    "$@"
