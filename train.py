from lightning.pytorch.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER

cli = LightningCLI(
    LitCoMER,
    CROHMEDatamodule,
    save_config_kwargs={"save_config_overwrite": True},
)
