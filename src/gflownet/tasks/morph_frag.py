import socket
from typing import Callable, Dict, List, Tuple, Union

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch_geometric.data import Data

from gflownet import FlatRewards, GFNTask, RewardScalar
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from gflownet.models import bengio2021flow, mmc
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward

from sklearn.metrics.pairwise import cosine_similarity


class MorphSimilarityTask(GFNTask):
    """Sets up a task where the reward is computed using the cosine distance between the latent
    morphology representation of a target molecule and the latent representation of a generated molecule
    """

    def __init__(
        self,
        target,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.target = self._setup_target_representation(target)
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        ckpt_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/morph_struct.ckpt"
        cfg_dir = "/home/mila/s/stephen.lu/gfn_gene/multimodal_contrastive/configs"
        cfg_name = "puma_sm_gmc.yaml"
        worker_device = get_worker_device()
        model = mmc.MMC_Proxy(cfg_name, cfg_dir, ckpt_path, worker_device).get_model()
        return {"mmc": model}

    def _setup_target_representation(self, target):
        # TODO: find a nicer way to add None batch axis to the 1D inputs
        target["inputs"]["morph"] = target["inputs"]["morph"][None, ...]
        target["inputs"]["joint"]["morph"] = target["inputs"]["joint"]["morph"][None, ...]
        pred = self.models["mmc"](target, mod_name="joint")
        return pred.cpu().detach().numpy()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        """
        Gets the latent representation of the molecules and computes the reward as the
        cosine distance between these latents and the target latent
        """
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["mmc"].device if hasattr(self.models["mmc"], "device") else get_worker_device())
        preds = self.models["mmc"]({"inputs": {"struct": batch}}, mod_name="struct").data.cpu().detach().numpy()
        preds = cosine_similarity(self.target, preds) + 1
        preds[np.isnan(preds)] = 0
        return self.flat_reward_transform(preds).clip(1e-4, 2).reshape((-1,))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [mmc.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return FlatRewards(preds), is_valid


class MorphSimilarityTrainer(StandardOnlineTrainer):
    task: MorphSimilarityTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        target_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample.pkl"
        with open(target_path, "rb") as f:
            target = pickle.load(f)

        assert "inputs" in target

        self.task = MorphSimilarityTask(
            target=target,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/toy_task"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.num_training_steps = 100
    config.validate_every = 0
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.num_workers = 0
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    trial = MorphSimilarityTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
