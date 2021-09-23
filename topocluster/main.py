from conduit.data.datamodules.vision import ColoredMNISTDataModule, WaterbirdsDataModule
from conduit.models.self_supervised import DINO, MoCoV2
from kit.hydra import Option

from topocluster.relays import SSLRelay

if __name__ == "__main__":
    SSLRelay.with_hydra(
        base_config_dir="conf",
        datamodules=[
            Option(ColoredMNISTDataModule, name="cmnist"),
            Option(WaterbirdsDataModule, name="waterbirds"),
        ],
        models=[
            Option(MoCoV2, name="moco"),
            Option(DINO, name="dino"),
        ],
    )
