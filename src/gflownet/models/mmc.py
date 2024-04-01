import numpy as np
import hydra
import wandb
from omegaconf import OmegaConf

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol as RDMol

from typing import Optional
from torch.nn import Module
from pytorch_lightning import LightningModule

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from multimodal_contrastive.utils import utils
from multimodal_contrastive.data.featurization import mol_to_data
from multimodal_contrastive.networks.utils import move_batch_input_to_device

OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list))


class MMC_Proxy(Module):
    model: Optional[Module]
    latents: Optional[np.ndarray]
    wrap_for_mp: bool = False

    def __init__(self, cfg_name: str, cfg_dir: str, ckpt_path: str, worker: str):
        super().__init__()
        with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = hydra.compose(config_name=cfg_name)
        model: LightningModule = utils.instantiate_model(cfg)
        model = model.load_from_checkpoint(ckpt_path, map_location=worker)
        model = model.to(worker)
        self.model = model.eval()

        # TODO make optional and load from arg path
        self.latents = np.load("/home/mila/s/stephen.lu/gfn_gene/res/mmc/puma_embeddings.npz")

    def log_target_properties(self, target, target_latent, mode="joint"):
        fig, ax = plt.subplots(figsize=(5, 5))

        # compute cosine similarity between target modality and all struct latents
        cosine_sim = cosine_similarity(target_latent, self.latents["struct"])
        ax.hist(cosine_sim.flatten(), bins=50)
        ax.set_title(f"Cosine similarity to target")

        if mode == "joint":
            mol = Chem.MolFromSmiles(bytes(target["inputs"]["struct"].mols))
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            # plot the target molecule
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            img = Draw.MolToImage(mol)
            ax2.imshow(img)
            ax2.axis("off")

            print(f"Target num atoms: {num_atoms}, Target num bonds: {num_bonds}")

        if not wandb.run:
            plt.clf()               
            return

        wandb.log({"Target properties": wandb.Image(fig)})

        if mode == "joint":
            wandb.log(
                {"Target num atoms": num_atoms, "Target num bonds": num_bonds, "Target molecule": wandb.Image(fig2)}
            )

    def get_model(self, wrap_model_fn=None):
        if wrap_model_fn is None:
            return self.model

        self.wrap_for_mp = True

        self.encoders = {k: wrap_model_fn(v) for (k, v) in self.model.encoders.items()}
        self.projectors = {k: wrap_model_fn(v) for k, v in self.model.projectors.items()}
        self.model.encoders = None
        self.model.projectors = None

        if self.model.__class__.__name__ == "GMC_PL":
            self.encoder_joint = self.encoders["joint"]
            self.common_encoder = wrap_model_fn(self.model.common_encoder)
            self.model.common_encoder = None

        elif self.model.__class__.__name__ == "CLIP_PL":
            pass

        return self

    def forward(self, batch, mod_name="struct"):
        x_dict = batch["inputs"]
        assert mod_name in ["struct", "morph", "joint", "ge"]
        return self._mod_encode(x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name])

    def _mod_encode(self, x_mod, encoders_mod, proj):
        if self.model.__class__.__name__ == "GMC_PL":
            return self.common_encoder(proj(encoders_mod(x_mod)))
        elif self.model.__class__.__name__ == "CLIP_PL":
            return proj(encoders_mod(x_mod))
        else:
            raise NotImplementedError


def to_device(x_dict, device=None):
    return move_batch_input_to_device(x_dict, device=device)


def mol2graph(mol: RDMol):
    return mol_to_data(mol, mode="mol")
