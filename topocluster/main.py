from conduit.data.datamodules.vision import ColoredMNISTDataModule
from ranzen.hydra.relay import Option

from topocluster.models.cp import ContrastivePersistence
from topocluster.relays import SSLRelay

if __name__ == "__main__":
    SSLRelay.with_hydra(
        root="conf",
        datamodule=[Option(ColoredMNISTDataModule, name="cmnist")],
        model=[Option(ContrastivePersistence, name="cp")],
        clear_cache=True,
    )
