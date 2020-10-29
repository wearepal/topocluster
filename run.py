from dataclasses import dataclass
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.omegaconf import MISSING

from gen.gudhi.clustering.tomato.conf import TomatoConf
from gen.topocluster.optimisation.unsupervised.kmeans.conf import KmeansConf
from topocluster.configs.arguments import ClustererConfig, PbcConfig

__all__ = ["main"]


@dataclass
class Config:
    # data: DataConfig
    clusterer: Any = MISSING


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="primary", node=Config)
cs.store(group="schema/clusterer", name="tomato", node=TomatoConf, package="clusterer")
cs.store(group="schema/clusterer", name="kmeans", node=KmeansConf, package="clusterer")


@hydra.main(config_path="conf", config_name="primary")
def main(cfg: Config) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    clusterer = instantiate(cfg.clusterer)
    print(clusterer)


if __name__ == "__main__":
    main()
