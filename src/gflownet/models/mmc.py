import numpy as np
import hydra
from omegaconf import OmegaConf
from rdkit.Chem.rdchem import Mol as RDMol
from pytorch_lightning import LightningModule

from multimodal_contrastive.utils import utils
from multimodal_contrastive.data.featurization import mol_to_data

OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list))


class MMC_Proxy:
    """Helper class that loads a multimodal contrastive model from a checkpoint"""

    def __init__(self, cfg_name: str, cfg_dir: str, ckpt_path: str, worker: str):
        with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = hydra.compose(config_name=cfg_name)
        model: LightningModule = utils.instantiate_model(cfg)
        model = model.load_from_checkpoint(ckpt_path, map_location=worker)
        model = model.to(worker)
        self.model = model.eval()

    def get_model(self):
        return self.model


def mol2graph(mol: RDMol):
    return mol_to_data(mol, mode="mol")
