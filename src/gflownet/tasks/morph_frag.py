import dill
import random
import socket
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
import wandb
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from gflownet.models import bengio2021flow, mmc
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.multiobjective_hooks import (
    AtomicPropertiesHook,
    NumberOfModesHook,
    NumberOfScaffoldsHook,
    NumberOfUniqueTrajectoriesHook,
    RewardPercentilesHook,
    SnapshotDistributionHook,
    TopSimilarityToTargetHook,
)
from gflownet.utils.transforms import to_logreward


class MorphSimilarityTask(GFNTask):
    """Sets up a task where the reward is computed using the cosine distance between the latent
    morphology representation of a target molecule and the latent representation of a generated molecule
    """

    def __init__(
        self,
        raw_target,
        dataset: Dataset,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.cfg = cfg
        self.models = self._load_task_models()

        self.target_mol = None
        if "struct" in raw_target["inputs"]:
            self.target_mol = Chem.MolFromSmiles(bytes(raw_target["inputs"]["struct"].mols))
            self.dataset = dataset
            self.dataset.set_smis([bytes(raw_target["inputs"]["struct"].mols)] * cfg.algo.num_from_dataset)

        struct_latent, morph_latent, joint_latent = self._setup_target_representation(raw_target)
        self.target = morph_latent if cfg.task.morph_sim.target_mode == "morph" else joint_latent
        self.mmc_proxy.log_target_properties(
            self.target_mol, struct_latent, morph_latent, joint_latent, mode=self.cfg.task.morph_sim.target_mode
        )
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        return ObjectProperties(torch.as_tensor(y))

    def inverse_reward_transform(self, rp):
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

        with torch.no_grad():
            morph_latent = self.models["mmc"](target, mod_name="morph").cpu().detach().numpy()
            struct_latent = self.models["mmc"](target, mod_name="struct").cpu().detach().numpy()
            joint_latent = self.models["mmc"](target, mod_name="joint").cpu().detach().numpy()

        return struct_latent, morph_latent, joint_latent

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

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
        return self.reward_transform(preds).clip(1e-4, 1).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [mmc.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid

    def get_model_latents(self, mols: List[RDMol]) -> Tensor:
        graphs = [mmc.mol2graph(i) for i in mols]
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["mmc"].device if hasattr(self.models["mmc"], "device") else get_worker_device())
        inputs = {"inputs": {"struct": batch}}
        preds = self.models["mmc"](inputs, mod_name="struct").data.cpu().detach().numpy()
        return preds


class TargetDataset(Dataset):
    """Note: this dataset isn't used by default, but turning it on showcases some features of this codebase.

    To turn on, self `cfg.algo.num_from_dataset > 0`"""

    def __init__(self, smis) -> None:
        super().__init__()
        self.props: List[Tensor] = []
        self.mols: List[Graph] = []
        self.smis = smis

    def set_smis(self, smis):
        self.smis = smis

    def setup(self, task: MorphSimilarityTask, ctx: FragMolBuildingEnvContext):
        rdmols = [Chem.MolFromSmiles(i) for i in self.smis]
        self.mols = [ctx.obj_to_graph(i) for i in rdmols]
        self.props = task.compute_obj_properties(rdmols)[0]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, index):
        return self.mols[index], self.props[index]


class MorphSimilarityTrainer(StandardOnlineTrainer):
    task: MorphSimilarityTask
    training_data: TargetDataset

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
            target = f.read()
            target = dill.loads(target)

        assert "inputs" in target

        self.task = MorphSimilarityTask(
            raw_target=target,
            dataset=self.training_data,
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        super().setup_data()
        self.training_data = TargetDataset([])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            # try using the smaller set of 18 fragments instead of the whole set
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.morph_sim.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)

        self.sampling_hooks.append(AtomicPropertiesHook())
        self.sampling_hooks.append(RewardPercentilesHook())
        self.sampling_hooks.append(NumberOfScaffoldsHook())
        self.sampling_hooks.append(NumberOfUniqueTrajectoriesHook())
        self.sampling_hooks.append(TopSimilarityToTargetHook(self.task.target_mol))

        self._num_modes_hooks = [
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.7, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.75, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.8, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.85, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.9, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.95, sim_thresholds=[0.7, 0.5, 0.2]),
            NumberOfModesHook(reward_mode="hard", reward_threshold=0.975, sim_thresholds=[0.7, 0.5, 0.2]),
        ]
        self.sampling_hooks.extend(self._num_modes_hooks)

        self._snapshot_dist_hook = SnapshotDistributionHook()
        self.valid_sampling_hooks.append(self._snapshot_dist_hook)

    def build_callbacks(self):
        parent = self
        callback_dict = {}

        class NumModesCallback:
            def on_validation_end(self, metrics):
                for hook in parent._num_modes_hooks:
                    if hook.should_stop_logging(hook.sim_high):
                        continue
                    simkey = hook.get_key_from_sim(hook.sim_high)
                    modes_by_scaffold = hook.split_by_scaffold()
                    num_to_plot = min(10, len(modes_by_scaffold))
                    scaffolds = random.sample(list(modes_by_scaffold.keys()), num_to_plot)
                    for scaffold in scaffolds:
                        mol, top_rew = modes_by_scaffold[scaffold][0]
                        fig, ax = mmc.plot_mol(mol, top_rew, scaffold)
                        metrics[f"{hook.__label__(simkey)}"] = wandb.Image(fig)
                        plt.close(fig)

        class SnapshotDistCallback:
            def on_validation_end(self, metrics):
                reward_fig = parent._snapshot_dist_hook.plot_reward_distribution()
                top_k_sim_fig = parent._snapshot_dist_hook.plot_top_k_tanimoto_similarity()
                metrics["reward_dist"] = wandb.Image(reward_fig)
                metrics["top_k_similarity_dist"] = wandb.Image(top_k_sim_fig)

                if parent.task.target_mol:
                    sim_to_target_fig = parent._snapshot_dist_hook.plot_tanimoto_similarity_to_target(
                        parent.task.target_mol
                    )
                    metrics["tanimoto_sim_to_target"] = wandb.Image(sim_to_target_fig)
                    plt.close(sim_to_target_fig)

                plt.close(reward_fig)
                plt.close(top_k_sim_fig)

        class WarmupTrajectoryLength:
            """Increments the maximum trajectory length over the course of training"""

            def on_validation_end(self, metrics):
                max_nodes = 8
                if parent.ctx.max_frags < max_nodes:
                    parent.ctx.max_frags += 1
                    parent.algo.graph_sampler.max_nodes = parent.ctx.max_frags

        # callback_dict["warmup_traj_len"] = WarmupTrajectoryLength()
        callback_dict["num_modes"] = NumModesCallback()
        callback_dict["snapshot_dist"] = SnapshotDistCallback()
        return callback_dict


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "~/scratch/morph_frag_logs"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.pickle_mp_messages = True
    config.num_training_steps = 100
    config.validate_every = 10
    config.num_validation_gen_steps = 5
    config.num_final_gen_steps = 100
    config.num_workers = 0
    config.opt.lr_decay = 2000
    config.algo.sampling_tau = 0.99

    config.algo.method = "TB"
    config.algo.max_nodes = 6
    config.algo.train_random_action_prob = 0.05
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]
    config.cond.temperature.num_thermometer_dim = 1

    config.algo.num_from_policy = 64
    config.algo.num_from_dataset = 0
    config.algo.valid_num_from_policy = 64
    config.algo.valid_num_from_dataset = 0

    config.replay.use = False
    config.replay.capacity = 1000
    config.replay.warmup = 100
    config.replay.num_from_replay = 32
    config.replay.num_new_samples = 32

    config.task.morph_sim.target_path = "path.to/sample.pkl"
    config.task.morph_sim.proxy_path = "path.to/mmc_proxy.ckpt"
    config.task.morph_sim.config_dir = "absolute.path.to/multimodal_contrastive/configs"
    config.task.morph_sim.config_name = "puma_sm_gmc.yaml"
    config.task.morph_sim.reduced_frag = False
    config.task.morph_sim.target_mode = "joint"

    trial = MorphSimilarityTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
