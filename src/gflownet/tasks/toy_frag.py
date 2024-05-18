import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward

TARGET_MOL = "O=C(NCc1cc(CCc2cccc(N3CCC(c4cc(-c5cc(-c6cncnc6)[nH]n5)ccn4)CC3)c2)ccn1)c1cccc2ccccc12"


class ToySimilarityTask(GFNTask):
    """Sets up a task where the reward is computed using the Tanimoto distance between
    the Morgan fingerprints of the generated molecule and a target molecule.
    """

    def __init__(
        self,
        target: str,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.target = Chem.MolFromSmiles(target)
        self.fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        self.target_fp = self.fpgen.GetSparseCountFingerprint(self.target)
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        return ObjectProperties(torch.as_tensor(y) / 4)

    def inverse_reward_transform(self, rp):
        return rp * 4

    def _load_task_models(self):
        return {}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        raise NotImplementedError

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        """
        Computes Tanimoto distance between morgan fingerprints of src and target mols
        we don't need to transform from RDMol to graph here since we can compute the
        naive reward directly in the molecule space
        """
        fps = [self.fpgen.GetSparseCountFingerprint(m) for m in mols]
        is_valid = torch.tensor([fp is not None for fp in fps]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = torch.as_tensor(
            [DataStructs.TanimotoSimilarity(fp, self.target_fp) for fp in fps if fp is not None]
        ).reshape(-1, 1)
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class ToySimilarityTrainer(StandardOnlineTrainer):
    task: ToySimilarityTask

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
        self.task = ToySimilarityTask(
            target=TARGET_MOL,
            cfg=self.cfg,
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
    config.num_training_steps = 10_000
    config.validate_every = 0
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.num_workers = 8
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]

    trial = ToySimilarityTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
