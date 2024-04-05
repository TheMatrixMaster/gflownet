import socket
from typing import Callable, Dict, List, Tuple, Union

import random
import wandb
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
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow, mmc
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward
from gflownet.utils.multiobjective_hooks import (
    AtomicPropertiesHook,
    RewardPercentilesHook,
    NumberOfModesHook,
    SnapshotDistributionHook,
    NumberOfUniqueTrajectoriesHook,
)

from sklearn.metrics.pairwise import cosine_similarity


class MorphSimilarityTask(GFNTask):
    """Sets up a task where the reward is computed using the cosine distance between the latent
    morphology representation of a target molecule and the latent representation of a generated molecule
    """

    def __init__(
        self,
        raw_target,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.models = self._load_task_models()
        self.target = self._setup_target_representation(raw_target)
        self.mmc_proxy.log_target_properties(raw_target, self.target, mode=self.cfg.task.morph_sim.target_mode)
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        ckpt_path = self.cfg.task.morph_sim.proxy_path
        cfg_dir = self.cfg.task.morph_sim.config_dir
        cfg_name = self.cfg.task.morph_sim.config_name
        self.mmc_proxy = mmc.MMC_Proxy(cfg_name, cfg_dir, ckpt_path, get_worker_device())

        if self.cfg.num_workers > 0:
            model = self.mmc_proxy.get_model(self._wrap_model)
        else:
            model = self._wrap_model(self.mmc_proxy.get_model())

        return {"mmc": model}

    def _setup_target_representation(self, target):
        # TODO: find a nicer way to add None batch axis to the 1D inputs
        target["inputs"]["morph"] = target["inputs"]["morph"][None, ...]
        target["inputs"]["joint"]["morph"] = target["inputs"]["joint"]["morph"][None, ...]
        target["inputs"] = mmc.to_device(target["inputs"], device=get_worker_device())
        assert self.cfg.task.morph_sim.target_mode in ["morph", "joint"], "Invalid target mode"
        pred = self.models["mmc"](target, mod_name=self.cfg.task.morph_sim.target_mode)
        # in reality, we may only have access to the morphology / transcriptomics of the target
        # but we may not always have access to a target molecular structure
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
        inputs = {"inputs": {"struct": batch}}

        preds = self.models["mmc"](inputs, mod_name="struct").data.cpu().detach().numpy()
        preds = (cosine_similarity(self.target, preds) + 1) / 2
        preds[np.isnan(preds)] = 0
        return self.flat_reward_transform(preds).clip(1e-4, 1).reshape((-1,))

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
        target_path = self.cfg.task.morph_sim.target_path
        with open(target_path, "rb") as f:
            target = pickle.load(f)

        assert "inputs" in target

        self.task = MorphSimilarityTask(
            raw_target=target,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            # try using the smaller set of 18 fragments instead of the whole set
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.morph_sim.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        # Creates the following sampling hooks:
        #
        # - Throughout training, saves the molecules with reward higher than a certain threshold
        #   (could be top-k percentile or a fixed threshold) and plot the number of unique modes
        #   where a mode is a new molecule with high reward whose tanimoto similarity with all
        #   the previous modes is below a certain threshold.
        super().setup()
        self.sampling_hooks.append(AtomicPropertiesHook())
        self.sampling_hooks.append(RewardPercentilesHook())
        self.sampling_hooks.append(NumberOfUniqueTrajectoriesHook())

        self._num_modes_hooks = [
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.7, sim_threshold=0.7),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.8, sim_threshold=0.7),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.9, sim_threshold=0.7),
        ]
        self.sampling_hooks.extend(self._num_modes_hooks)

        self._snapshot_dist_hook = SnapshotDistributionHook()
        self.valid_sampling_hooks.append(self._snapshot_dist_hook)

    def build_callbacks(self):
        # Creates the following callback classes that implement `on_validation_end(self, metrics: Dict[str, Any])`:
        #
        # - Samples X new trajectories during validation, keep the top-k molecules with the highest
        #   reward and plot a histogram of the tanimoto similarity between the top-k molecules and
        #   the target molecule.
        #
        # - Samples X new trajectories during validation and plot their reward distribution
        #
        # - Uses the high reward molecules saved during training, plot the top k molecules with
        #   highest reward that have different Murcko scaffolds
        parent = self
        callback_dict = {}

        class NumModesCallback:
            def on_validation_end(self, metrics):
                for hook in parent._num_modes_hooks:
                    modes_by_scaffold = hook.split_by_scaffold()
                    num_to_plot = min(10, len(modes_by_scaffold))
                    scaffolds = random.sample(list(modes_by_scaffold.keys()), num_to_plot)
                    for scaffold in scaffolds:
                        mol, top_rew = modes_by_scaffold[scaffold][0]
                        fig, ax = mmc.plot_mol(mol)
                        metrics[f"reward: {top_rew}, scaffold: {scaffold}"] = wandb.Image(fig)
                        fig.clear()

        class SnapshotDistCallback:
            def on_validation_end(self, metrics):
                reward_fig = parent._snapshot_dist_hook.plot_reward_distribution()
                top_k_sim_fig = parent._snapshot_dist_hook.plot_top_k_tanimoto_similarity()
                metrics["reward_dist"] = wandb.Image(reward_fig)
                metrics["top_k_similarity_dist"] = wandb.Image(top_k_sim_fig)
                reward_fig.clear()
                top_k_sim_fig.clear()

        callback_dict["num_modes"] = NumModesCallback()
        callback_dict["snapshot_dist"] = SnapshotDistCallback()
        return callback_dict


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/toy_task"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.pickle_mp_messages = True
    config.num_training_steps = 1000
    config.validate_every = 5
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 1000
    config.num_workers = 0
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.95
    config.algo.max_nodes = 7
    config.algo.train_random_action_prob = 0.01
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]
    config.cond.temperature.num_thermometer_dim = 1

    config.replay.use = False
    config.replay.capacity = 1000
    config.replay.warmup = 100

    config.algo.num_from_policy = 64
    config.replay.num_from_replay = 32
    config.replay.num_new_samples = 32

    config.task.morph_sim.target_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_0.pkl"
    config.task.morph_sim.proxy_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/morph_struct_90_step_val_loss.ckpt"
    config.task.morph_sim.config_dir = "/home/mila/s/stephen.lu/gfn_gene/multimodal_contrastive/configs"
    config.task.morph_sim.config_name = "puma_sm_gmc.yaml"
    config.task.morph_sim.reduced_frag = False
    config.task.morph_sim.target_mode = "joint"

    trial = MorphSimilarityTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
