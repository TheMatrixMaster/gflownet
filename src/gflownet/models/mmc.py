from collections import defaultdict
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from multimodal_contrastive.data.featurization import mol_to_data
from multimodal_contrastive.networks.utils import move_batch_input_to_device
from multimodal_contrastive.utils import utils
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol as RDMol
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import Module

OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list))


class MMC_Proxy(Module):
    model: Optional[Module]
    wrap_for_mp: bool = False

    def __init__(self, cfg_name: str, cfg_dir: str, ckpt_path: str, worker: str):
        super().__init__()
        with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = hydra.compose(config_name=cfg_name)
        model: LightningModule = utils.instantiate_model(cfg)
        model = model.load_from_checkpoint(ckpt_path, map_location=worker)
        model = model.to(worker)
        self.model = model.eval()
        self.cache = defaultdict(float)

    def log_target_properties(self, target_mol, struct_latent, morph_latent, joint_latent, mode="morph"):
        struct_morph_sim = cosine_similarity(struct_latent, morph_latent)[0][0]
        struct_joint_sim = cosine_similarity(struct_latent, joint_latent)[0][0]
        morph_joint_sim = cosine_similarity(morph_latent, joint_latent)[0][0]

        print("Cosine similarity struct~morph: ", struct_morph_sim)
        print("Cosine similarity struct~joint: ", struct_joint_sim)
        print("Cosine similarity morph~joint: ", morph_joint_sim)

        if not wandb.run:
            return

        wandb.log(
            {
                "Cosine similarity struct~morph": struct_morph_sim,
                "Cosine similarity struct~joint": struct_joint_sim,
                "Cosine similarity morph~joint": morph_joint_sim,
            }
        )

        if target_mol:
            mol = target_mol
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            # plot the target molecule
            fig2, ax2 = plot_mol(mol)
            print(f"Target num atoms: {num_atoms}, Target num bonds: {num_bonds}")

            wandb.log(
                {"Target num atoms": num_atoms, "Target num bonds": num_bonds, "Target molecule": wandb.Image(fig2)}
            )

        plt.clf()

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


def plot_mol(mol: RDMol, top_rew: float = None, scaffold: str = None):
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    # make title
    title = f"Num atoms: {num_atoms}, Num bonds: {num_bonds}"
    if top_rew:
        title += f"\n Reward: {top_rew}"
    if scaffold:
        title += f"\n Scaffold: {scaffold}"

    # plot the target molecule
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    img = Draw.MolToImage(mol)
    ax.imshow(img)
    ax.axis("off")

    return fig, ax
