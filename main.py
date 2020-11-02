import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gen.gudhi.clustering.tomato.conf import TomatoConf
from gen.pytorch_lightning.conf import TrainerConf
from gen.topocluster.clustering.kmeans.conf import KmeansConf
from gen.topocluster.data.data_modules.conf import MNISTDataModuleConf
from gen.topocluster.experiment.conf import ExperimentConf
from gen.topocluster.models.autoencoder.conf import GatedConvAutoEncoderConf


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConf)
cs.store(group="schema/datamodule", name="mnist", node=MNISTDataModuleConf, package="datamodule")
cs.store(group="schema/encoder", name="gconv_ae", node=GatedConvAutoEncoderConf, package="encoder")
cs.store(group="schema/clusterer", name="tomato", node=TomatoConf, package="clusterer")
cs.store(group="schema/clusterer", name="kmeans", node=KmeansConf, package="clusterer")
# cs.store(group="schema/trainer", name="trainer", node=TrainerConf, package="trainer")
# cs.store(group="schema/trainer", name="pt_trainer", node=TrainerConf, package="trainer")


@hydra.main(config_path="conf", config_name="experiment")
def launcher(cfg: ExperimentConf) -> None:
    exp = instantiate(cfg, _recursive_=True)
    exp.start(OmegaConf.to_container(cfg))


if __name__ == "__main__":
    launcher()
