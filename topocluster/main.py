from conduit.data.datamodules.vision import ColoredMNISTDataModule, WaterbirdsDataModule
from conduit.models.self_supervised import DINO, MoCoV2
from ranzen.hydra.relay import Option

from topocluster.relays import SSLRelay

if __name__ == "__main__":
    SSLRelay.with_hydra(
        root="conf",
        datamodule=[
            Option(ColoredMNISTDataModule, name="cmnist"),
            Option(WaterbirdsDataModule, name="waterbirds"),
        ],
        model=[
            Option(MoCoV2, name="moco"),
            Option(DINO, name="dino"),
        ],
    )
