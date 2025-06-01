"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod
from typing import Callable, Optional, Union

import torch
from loguru import logger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_warn
from torch.optim.lr_scheduler import _LRScheduler

from nndet.training.learning_rate import CycleLinear

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


def _get_default_scheduler_config():
    """Default scheduler configuration for PyTorch Lightning compatibility."""
    return {
        "scheduler": None,
        "interval": "epoch",
        "frequency": 1,
        "monitor": None,
        "strict": True,
        "name": None,
    }


class BaseSWA(StochasticWeightAveraging):
    def __init__(
        self,
        swa_epoch_start: int,
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        update_statistics: Optional[bool] = False,
    ):
        """
        New Base Class for Stochastic Weighted Averaging

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
                
        Notes: Does not support updating of norm weights after training
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            swa_lrs=0.01,  # Provide a default positive value (will be overridden by custom scheduler)
            annealing_epochs=10,
            annealing_strategy="cos",
            avg_fn=avg_fn,
            device=device,
        )
        self.update_statistics = update_statistics
        logger.info(f"Initialize SWA with swa epoch start {self.swa_start}")

    def pl_module_contains_batch_norm(self, pl_module: 'pl.LightningModule'):
        if self.update_statistics:
            raise NotImplementedError("Updating the statistis of the "
                                      "normalization layer is not suported yet.")
        else:
            return self.update_statistics

    def on_train_epoch_start(self,
                             trainer: 'pl.Trainer',
                             pl_module: 'pl.LightningModule',
                             ):
        """
        Repalce current lr scheduler with SWA scheduler
        """
        if trainer.current_epoch == self.swa_start:
            # PyTorch Lightning v2.0 compatibility: use strategy to get optimizers
            if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'optimizers'):
                optimizers = trainer.strategy.optimizers
            else:
                optimizers = trainer.optimizers
            
            if isinstance(optimizers, list) and len(optimizers) > 0:
                optimizer = optimizers[0]
            else:
                # Fallback to pl_module's optimizers
                opt_config = pl_module.configure_optimizers()
                if isinstance(opt_config, tuple) and len(opt_config) >= 1:
                    optimizer = opt_config[0][0] if isinstance(opt_config[0], list) else opt_config[0]
                else:
                    optimizer = opt_config

            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

            _scheduler = self.get_swa_scheduler(optimizer)
            
            # Store the actual scheduler object, not a config dict
            if isinstance(_scheduler, dict):
                self._swa_scheduler = _scheduler["scheduler"]
                scheduler_config = _scheduler
            else:
                self._swa_scheduler = _scheduler
                scheduler_config = {"scheduler": _scheduler}

            # PyTorch Lightning v2.0 compatibility: use lr_scheduler_configs instead of lr_schedulers
            if hasattr(trainer, 'lr_scheduler_configs'):
                lr_scheduler_configs = trainer.lr_scheduler_configs
            else:
                # Fallback for older versions
                lr_scheduler_configs = getattr(trainer, 'lr_schedulers', [])

            if lr_scheduler_configs:
                if hasattr(trainer, 'lr_scheduler_configs'):
                    lr_scheduler = lr_scheduler_configs[0].scheduler
                else:
                    lr_scheduler = lr_scheduler_configs[0]["scheduler"]
                rank_zero_warn(f"Swapping lr_scheduler {lr_scheduler} for {self._swa_scheduler}")
                if hasattr(trainer, 'lr_scheduler_configs'):
                    # For v2.0: need to update the actual config object
                    trainer.lr_scheduler_configs[0].scheduler = self._swa_scheduler
                    trainer.lr_scheduler_configs[0].interval = scheduler_config.get("interval", "epoch")
                    trainer.lr_scheduler_configs[0].frequency = scheduler_config.get("frequency", 1)
                else:
                    trainer.lr_schedulers[0] = scheduler_config
            else:
                # For new schedulers, let the trainer handle the configuration
                # Simply set the optimizer with the scheduler - trainer will configure it
                pass

            self.n_averaged = torch.tensor(0, dtype=torch.long, device=pl_module.device)

        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self.avg_fn)

        if trainer.current_epoch == self.swa_end + 1:
            raise NotImplementedError("This should never happen (yet)")

    @abstractmethod
    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        """
        Generate LR scheduler for SWA

        Args:
            optimizer: optimizer to wrap

        Returns:
            Union[_LRScheduler, dict]: If a lr scheduler is returned it will
                be stepped once per epoch. Can also return a whole config of
                the scheduler to customize steps.
        """
        raise NotImplementedError


class SWACycleLinear(BaseSWA):
    def __init__(self,
                 swa_epoch_start: int,
                 cycle_initial_lr: float,
                 cycle_final_lr: float,
                 num_iterations_per_epoch: int,
                 avg_fn: Optional[_AVG_FN] = None,
                 device: Optional[Union[torch.device, str]] = torch.device("cpu"),
                 update_statistics: Optional[bool] = None,
                 ):
        """
        SWA based on :class:`CycleLinear`

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            cycle_initial_lr: initial learning rate of cycle
            cycle_final_lr: final learning rate of cycle
            num_iterations_per_epoch: number of train iterations per epoch
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            avg_fn=avg_fn,
            device=device,
            update_statistics=update_statistics,
            )
        self.cycle_initial_lr = cycle_initial_lr
        self.cycle_final_lr = cycle_final_lr
        self.num_iterations_per_epoch = num_iterations_per_epoch

    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        return {
            "scheduler": CycleLinear(
                optimizer=optimizer,
                cycle_num_iterations=self.num_iterations_per_epoch,
                cycle_initial_lr=self.cycle_initial_lr,
                cycle_final_lr=self.cycle_final_lr,
                ),
            "interval": "step",
        }
