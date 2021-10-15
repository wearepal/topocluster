from __future__ import annotations
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import attr
from conduit.data.datamodules import CdtDataModule
from conduit.models import CdtModel
from hydra.utils import to_absolute_path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ranzen.hydra.relay import Option, Relay

__all__ = [
    "SSLRelay",
]


@attr.define(kw_only=True)
class SSLRelay(Relay):
    artifacts_dir: ClassVar[Path] = Path("artifacts")
    log_offline: bool = False
    exp_group: Optional[str] = None

    datamodule: CdtDataModule
    trainer: pl.Trainer
    model: CdtModel
    seed: Optional[int] = 42

    @classmethod
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        datamodule: List[Union[Type[Any], Option]],
        model: List[Union[Type[Any], Option]],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            datamodule=datamodule,
            model=model,
            trainer=[Option(class_=pl.Trainer, name="trainer")],
        )
        super().with_hydra(root=root, clear_cache=clear_cache, **configs)

    def run(self, raw_config: Dict[str, Any] | None = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")

        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        self.log(f"Artifacts directory: '{self.artifacts_dir.resolve()}'")

        logger_kwargs = dict(
            entity="predictive-analytics-lab",
            project="topocluster",
            offline=self.log_offline,
            group=self.model.__class__.__name__ if self.exp_group is None else self.exp_group,
        )
        train_logger = WandbLogger(**logger_kwargs, reinit=True)
        hparams = {"artifacts_dir": self.artifacts_dir.resolve(), "cwd": os.getcwd()}
        if raw_config is not None:
            self.log("-----\n" + str(raw_config) + "\n-----")
            hparams.update(raw_config)
        train_logger.log_hyperparams(hparams)

        if hasattr(self.datamodule, "root"):
            self.datamodule.root = to_absolute_path(self.datamodule.root)  # type: ignore
        self.datamodule.setup()
        self.datamodule.prepare_data()

        self.trainer.logger = train_logger
        self.model.run(datamodule=self.datamodule, trainer=self.trainer, seed=self.seed, copy=False)
        # Manually invoke finish for multirun-compatibility
        train_logger.experiment.finish()
