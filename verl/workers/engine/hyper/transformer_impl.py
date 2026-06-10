# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HyperParallel-backed transformer engines.

This backend intentionally reuses verl's FSDP HF-model training surface and swaps
the model wrapper to HyperParallel HSDP. It is a small integration layer for
RL smoke tests and incremental optimization, not a checkpoint-format migration.
"""

import logging
import os
import sys
import types
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions

from verl.utils.checkpoint.hyper_checkpoint_manager import HyperCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPOptimizerConfig, HFModelConfig, HyperEngineConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngine,
    FSDPEngineWithLMHead,
    FSDPEngineWithValueHead,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _ensure_hyper_parallel_importable() -> None:
    """Prefer an installed hyper_parallel, fallback to the sibling checkout."""
    os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
    try:
        import hyper_parallel  # noqa: F401

        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[5]
    sibling = repo_root.parent / "hyper-parallel"
    if sibling.exists():
        sys.path.insert(0, str(sibling))


_ensure_hyper_parallel_importable()


def _wrap_optimizer_with_skip_dtensor_dispatch(optimizer) -> None:
    """HyperParallel DTensor params need optimizer.step under SkipDTensorDispatch."""
    if optimizer is None or getattr(optimizer, "_verl_hyper_step_wrapped", False):
        return

    from hyper_parallel import SkipDTensorDispatch

    original_step = optimizer.step

    def _step(bound_optimizer, *args, **kwargs):
        del bound_optimizer
        with SkipDTensorDispatch():
            return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(_step, optimizer)
    setattr(optimizer, "_verl_hyper_step_wrapped", True)


class HyperEngine(FSDPEngine):
    """FSDP-compatible engine using HyperParallel's HSDP wrapper."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: HyperEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config,
    ):
        super().__init__(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )

    def initialize(self):
        self._build_model_optimizer()
        self.checkpoint_manager = HyperCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )
        log_gpu_memory_usage("After HyperParallel init", logger=logger)

    def _build_fsdp_module(self, module):
        from hyper_parallel import fully_shard, init_device_mesh
        from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
            reduce_dtype = torch.float32

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
            apply_grad_on_fp32_main_grad=self.engine_config.apply_grad_on_fp32_main_grad,
        )

        offload_policy = OffloadPolicy()
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            offload_policy = CPUOffloadPolicy(pin_memory=True)
            self._is_offload_param = False
            self._is_offload_optimizer = False

        world_size = torch.distributed.get_world_size()
        mesh = init_device_mesh(device_type=get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

        if self.engine_config.wrap_root_only:
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=self.engine_config.reshard_after_forward,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                comm_fusion=self.engine_config.comm_fusion,
                comm_fusion_zero_copy=self.engine_config.comm_fusion_zero_copy,
            )
        else:
            leaf_modules = [
                child
                for child in module.modules()
                if child is not module and len(list(child.children())) == 0 and sum(p.numel() for p in child.parameters()) > 0
            ]
            if leaf_modules:
                fully_shard(
                    leaf_modules,
                    mesh=mesh,
                    reshard_after_forward=self.engine_config.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    comm_fusion=self.engine_config.comm_fusion,
                    comm_fusion_zero_copy=self.engine_config.comm_fusion_zero_copy,
                )
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=self.engine_config.reshard_after_forward,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                comm_fusion=self.engine_config.comm_fusion,
                comm_fusion_zero_copy=self.engine_config.comm_fusion_zero_copy,
            )

        return module

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        _wrap_optimizer_with_skip_dtensor_dispatch(self.optimizer)

    def optimizer_step(self):
        from hyper_parallel.core.utils import clip_grad_norm_

        assert self.optimizer_config.clip_grad is not None
        grad_norm = clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)

        if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        if isinstance(grad_norm, torch.Tensor):
            return grad_norm.item()
        return float(grad_norm)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # HyperParallel HSDP owns param residency through its scheduler. Explicit
        # verl FSDP CPU/GPU offload helpers do not understand Hyper DTensors.
        super(FSDPEngine, self).to(device=device, model=model, optimizer=optimizer, grad=grad)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        del layered_summon, base_sync_done, kwargs

        from hyper_parallel.core.fully_shard.api import get_model_state_dict

        log_gpu_memory_usage("Before HyperParallel full state dict", logger=logger)
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        params = get_model_state_dict(self.module, options=options)
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
        log_gpu_memory_usage("After HyperParallel full state dict", logger=logger)

        device = get_device_id()
        per_tensor_param = (
            (name, param.to(device, non_blocking=True).to(torch.bfloat16, non_blocking=True))
            if isinstance(param, torch.Tensor)
            else (name, param)
            for name, param in params.items()
        )
        return per_tensor_param, None

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()

    def load_checkpoint(
        self, local_path: str, hdfs_path: str | None = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        del kwargs
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        torch.distributed.barrier()
        _wrap_optimizer_with_skip_dtensor_dispatch(self.optimizer)


@EngineRegistry.register(model_type="language_model", backend="hyper", device=["cuda", "npu"])
class HyperEngineWithLMHead(HyperEngine, FSDPEngineWithLMHead):
    pass


@EngineRegistry.register(model_type="value_model", backend="hyper", device=["cuda", "npu"])
class HyperEngineWithValueHead(HyperEngine, FSDPEngineWithValueHead):
    pass
