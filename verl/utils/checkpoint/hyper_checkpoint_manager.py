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
"""Checkpoint manager for the experimental HyperParallel engine."""

import json
import logging
import os
from dataclasses import asdict, dataclass

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions
from transformers import GenerationConfig
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import get_device_id, get_device_name
from verl.utils.fs import copy_to_local, local_mkdir_safe
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class HyperConfig:
    """Runtime metadata saved next to HyperParallel checkpoints."""

    world_size: int
    format: str = "hyper_full_state_dict_v1"


class HyperCheckpointManager(BaseCheckpointManager):
    """Save and load HyperParallel model, optimizer, scheduler, and RNG state.

    This is intentionally a lightweight checkpoint format for the verl adapter.
    It stores a full HyperParallel model state dict per rank. Hyper's
    ``load_state_dict`` can then distribute full tensors into local shards.
    """

    def __init__(self, *args, trust_remote_code: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_remote_code = trust_remote_code

    def _model_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"hyper_model_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _optim_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"hyper_optim_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _extra_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"hyper_extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _unwrap_model(self):
        return getattr(self.model, "_fsdp_wrapped_module", self.model)

    def _load_map_location(self) -> str:
        return f"{get_device_name()}:{get_device_id()}"

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        del hdfs_path, del_local_after_load
        if local_path is None:
            return

        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
            model_path = copy_to_local(self._model_path(local_path))
            model_state_dict = torch.load(model_path, map_location=self._load_map_location(), weights_only=False)
            self.model.load_state_dict(model_state_dict)
            log_with_rank(f"Loaded Hyper model from {model_path}", rank=self.rank, logger=logger)

        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )
            optim_path = copy_to_local(self._optim_path(local_path))
            optimizer_state_dict = torch.load(optim_path, map_location=self._load_map_location(), weights_only=False)
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(f"Loaded Hyper optimizer from {optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            extra_path = copy_to_local(self._extra_path(local_path))
            extra_state_dict = torch.load(extra_path, map_location="cpu", weights_only=False)
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {extra_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler")
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {extra_path}", rank=self.rank, logger=logger)

        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        del hdfs_path
        if local_path is None:
            return

        self.previous_global_step = global_step
        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        from hyper_parallel.core.fully_shard.api import get_model_state_dict

        model_state_dict = None
        if self.should_save_model:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state_dict = get_model_state_dict(self.model, options=options)
            model_path = self._model_path(local_path)
            torch.save(model_state_dict, model_path)
            log_with_rank(f"Saved Hyper model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

        if self.should_save_optimizer:
            optim_path = self._optim_path(local_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            log_with_rank(f"Saved Hyper optimizer to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

        if self.should_save_extra:
            extra_path = self._extra_path(local_path)
            lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            extra_state_dict = {
                "lr_scheduler": lr_scheduler_state_dict,
                "rng": self.get_rng_state(),
            }
            torch.save(extra_state_dict, extra_path)
            log_with_rank(f"Saved Hyper extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            unwrap_model = self._unwrap_model()
            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)

            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            hyper_config_path = os.path.join(local_path, "hyper_config.json")
            with open(hyper_config_path, "w") as f:
                json.dump(asdict(HyperConfig(world_size=self.world_size)), f, indent=4)

            if self.should_save_hf_model:
                if model_state_dict is None:
                    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                    model_state_dict = get_model_state_dict(self.model, options=options)
                hf_state_dict = {
                    key: value.to(torch.bfloat16) if isinstance(value, torch.Tensor) else value
                    for key, value in model_state_dict.items()
                }
                unwrap_model.save_pretrained(hf_config_tokenizer_path, state_dict=hf_state_dict)

        torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
