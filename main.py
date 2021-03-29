import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from gen.gudhi.clustering.tomato.conf import TomatoConf
from gen.pytorch_lightning.conf import TrainerConf
from gen.topocluster.clustering.dac.conf import PlClustererConf
from gen.topocluster.clustering.kmeans.conf import KmeansConf
from gen.topocluster.data.datamodules.conf import UMNISTDataModuleConf
from gen.topocluster.experiment.conf import ExperimentConf
from gen.topocluster.models.autoencoder.conf import ConvAutoEncoderConf
from kit import SchemaRegistration

sr = SchemaRegistration()
sr.register(path="experiment_schema", config_class=ExperimentConf)

# Definne the 'datamodule' group
with sr.new_group(group_name="schema/datamodule", target_path="datamodule") as group:
    group.add_option(name="umnist", config_class=UMNISTDataModuleConf)

# Definne the 'encoder' group
with sr.new_group(group_name="schema/encoder", target_path="encoder") as group:
    group.add_option(name="conv_ae", config_class=ConvAutoEncoderConf)

with sr.new_group(group_name="schema/clusterer", target_path="clusterer") as group:
    group.add_option(name="tomato", config_class=TomatoConf)
    group.add_option(name="kmeans", config_class=KmeansConf)
    group.add_option(name="plc", config_class=PlClustererConf)

# Definne the 'trainer'/'pretrainer' groups - these are singleton (containing one schema) groups
with sr.new_group(group_name="schema/trainer", target_path="trainer") as group:
    group.add_option(name="trainer", config_class=TrainerConf)
with sr.new_group(group_name="schema/pretrainer", target_path="pretrainer") as group:
    group.add_option(name="pretrainer", config_class=TrainerConf)


@hydra.main(config_path="conf", config_name="experiment")
def launcher(cfg: ExperimentConf) -> None:
    cfg.datamodule.data_dir = to_absolute_path(cfg.datamodule.data_dir)
    exp = instantiate(cfg, _recursive_=True)
    exp.start(OmegaConf.to_container(cfg))


if __name__ == "__main__":
    launcher()
