from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import attr
from conduit.relays import CdtRelay
from pytorch_lightning.loggers import WandbLogger

__all__ = [
    "SSLRelay",
]


@attr.define(kw_only=True)
class SSLRelay(CdtRelay):
    artifacts_dir: ClassVar[Path] = Path("artifacts")

    log_offline: bool = False
    exp_group: Optional[str] = None

    def run(self, raw_config: dict[str, Any] | None = None):
        self.datamodule.setup()
        self.datamodule.prepare_data()
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        print(f"Current working directory: '{os.getcwd()}'")
        print(f"Artifacts directory: '{self.artifacts_dir.resolve()}'")

        logger_kwargs = dict(
            entity="predictive-analytics-lab",
            project="topocluster",
            offline=self.log_offline,
            group=self.model.__class__.__name__ if self.exp_group is None else self.exp_group,
        )
        train_logger = WandbLogger(**logger_kwargs, reinit=True)
        hparams = {"artifacts_dir": self.artifacts_dir.resolve(), "cwd": os.getcwd()}
        if raw_config is not None:
            print("-----\n" + str(raw_config) + "\n-----")
            hparams.update(raw_config)
        train_logger.log_hyperparams(hparams)
        self.trainer.logger = train_logger
        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)
        # Manually invoke finish for multirun-compatibility
        train_logger.experiment.finish()
