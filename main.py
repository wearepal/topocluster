import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from gen.pytorch_lightning.trainer.conf import TrainerConf
from gen.topocluster.clustering.conf import (
    AgglomerativeClusteringConf,
    GaussianMixtureConf,
    KmeansConf,
    SpectralClusteringConf,
    TomatoConf,
    TopoGradConf,
)
from gen.topocluster.data.datamodules.conf import (
    MNISTDataModuleConf,
    UMNISTDataModuleConf,
)
from gen.topocluster.data.sampling.conf import GreedyCoreSetSamplerConf
from gen.topocluster.experiment.conf import ExperimentConf
from gen.topocluster.models.conf import (
    ConvAutoEncoderConf,
    ConvAutoEncoderMNISTConf,
    LeNet4Conf,
)
from gen.topocluster.reduction.conf import NoReduceConf, RandomProjectorConf, UMAPConf
from kit import SchemaRegistration


sr = SchemaRegistration()
sr.register(path="experiment_schema", config_class=ExperimentConf)

# Definne the 'datamodule' group
with sr.new_group(group_name="schema/datamodule", target_path="datamodule") as group:
    group.add_option(name="mnist", config_class=MNISTDataModuleConf)
    group.add_option(name="umnist", config_class=UMNISTDataModuleConf)

# Definne the 'encoder' group
with sr.new_group(group_name="schema/encoder", target_path="encoder") as group:
    group.add_option(name="conv_ae", config_class=ConvAutoEncoderConf)
    group.add_option(name="conv_ae_mnist", config_class=ConvAutoEncoderMNISTConf)
    group.add_option(name="lenet4", config_class=LeNet4Conf)

with sr.new_group(group_name="schema/clusterer", target_path="clusterer") as group:
    group.add_option(name="kmeans", config_class=KmeansConf)
    group.add_option(name="topograd", config_class=TopoGradConf)
    group.add_option(name="tomato", config_class=TomatoConf)
    group.add_option(name="agglom", config_class=AgglomerativeClusteringConf)
    group.add_option(name="gmm", config_class=GaussianMixtureConf)
    group.add_option(name="spectral", config_class=SpectralClusteringConf)
    # group.add_option(name="plc", config_class=PlClustererConf)

with sr.new_group(group_name="schema/reducer", target_path="reducer") as group:
    group.add_option(name="umap", config_class=UMAPConf)
    group.add_option(name="none", config_class=NoReduceConf)
    group.add_option(name="rand", config_class=RandomProjectorConf)

with sr.new_group(group_name="schema/sampler", target_path="sampler") as group:
    group.add_option(name="greedy", config_class=GreedyCoreSetSamplerConf)

# Definne the 'trainer'/'pretrainer' groups - these are singleton (containing one schema) groups
with sr.new_group(group_name="schema/trainer", target_path="trainer") as group:
    group.add_option(name="trainer", config_class=TrainerConf)
with sr.new_group(group_name="schema/pretrainer", target_path="pretrainer") as group:
    group.add_option(name="pretrainer", config_class=TrainerConf)


@hydra.main(config_path="conf", config_name="experiment")
def launcher(cfg: ExperimentConf) -> None:
    print(f"Current working directory: f{os.getcwd()}")
    cfg.datamodule.data_dir = to_absolute_path(cfg.datamodule.data_dir)
    exp = instantiate(cfg, _recursive_=True)
    exp.start(OmegaConf.to_container(cfg, enum_to_str=True))


if __name__ == "__main__":
    launcher()
