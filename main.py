import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gen.gudhi.clustering.tomato.conf import TomatoConf
from gen.topocluster.clustering.kmeans.conf import KmeansConf
from topocluster.confs import TrainConf
from topocluster.optimisation.train import main

__all__ = ["main"]


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="primary", node=TrainConf)
cs.store(group="schema/clusterer", name="tomato", node=TomatoConf, package="clusterer")
cs.store(group="schema/clusterer", name="kmeans", node=KmeansConf, package="clusterer")


@hydra.main(config_path="conf", config_name="primary")
def launcher(cfg: TrainConf) -> None:
    OmegaConf.to_yaml(cfg)
    clusterer = instantiate(cfg.clusterer)
    print(clusterer)


if __name__ == "__main__":
    launcher()
