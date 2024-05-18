import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext, generate_forward_trajectory
from gflownet.trainer import GFNAlgorithm
from gflownet.utils.misc import get_worker_device


class RandomSampling(GFNAlgorithm):
    """Create an algorithm that randomly samples trajectories from the environment."""

    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Instantiate model.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        cfg: Config
            Hyperparameters
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.model_is_autoregressive = False
        self.random_action_prob = [1, 1]

        self.graph_sampler = GraphSampler(
            ctx,
            env,
            cfg.algo.max_len,
            cfg.algo.max_nodes,
        )

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by randomly sampling the action space

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        n: int
            Number of trajectories to sample
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob=1.0)
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        return batch

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        """Computes a dummy loss value"""
        invalid_mask = 1 - batch.is_valid
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
        info = {
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "avg_number_of_nodes": np.mean([x.num_nodes for x in batch[final_graph_idx]]),
            "avg_number_of_edges": np.mean([x.num_edges for x in batch[final_graph_idx]]),
        }
        placeholder_loss = torch.tensor(0.0, device=batch.x.device, requires_grad=True)
        return placeholder_loss, info
