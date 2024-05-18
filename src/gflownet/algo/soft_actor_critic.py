import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.trainer import GFNAlgorithm
from gflownet.utils.misc import get_worker_device
from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)


class SoftActorCritic(GFNAlgorithm):
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Soft Actor Critic implementation, see
        Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. "Off-Policy Maximum Entropy Deep Reinforcement
        Learning with a Stochastic Actor" In International conference on machine learning, PMLR 80, 2018.

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        cfg: Config
            The experiment configuration
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.gamma = 1.0
        self.invalid_penalty = -75
        self.bootstrap_own_reward = False
        # we used fixed entropy regularization coefficient for now, but it is common to learn this
        self.alpha = 0.15
        # Experimental flags
        self.sample_temp = 1
        self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, self.sample_temp)

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float = 0
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
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
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
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

    def compute_batch_losses(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        critic1_target: nn.Module,
        critic2_target: nn.Module,
        opt_critic1,
        opt_critic2,
        batch: gd.Batch,
        num_bootstrap: int = 0,
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        actor: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        critic1: nn.Module
        critic2: nn.Module
        critic1_target: nn.Module
        critic2_target: nn.Module
          Two critic networks that learn Q(s,a) and their target networks
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = batch.log_rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        policy: GraphActionCategorical
        qf1_next_target: GraphActionCategorical
        qf2_next_target: GraphActionCategorical
        qf1_next: GraphActionCategorical
        qf2_next: GraphActionCategorical

        actor = actor.to(dev)
        critic1 = critic1.to(dev)
        critic2 = critic2.to(dev)
        critic1_target = critic1_target.to(dev)
        critic2_target = critic2_target.to(dev)

        # Critic Training
        with torch.no_grad():
            policy, _ = actor(batch, cond_info[batch_idx])
            next_state_log_pi = policy.logsoftmax()
            next_state_action_probs = [torch.exp(lp) for lp in next_state_log_pi]

            qf1_next_target, _ = critic1_target(batch, cond_info[batch_idx])
            qf2_next_target, _ = critic2_target(batch, cond_info[batch_idx])
            min_qf_next_target = [
                w * (torch.min(qf1, qf2) - self.alpha * lp)
                for w, qf1, qf2, lp in zip(
                    next_state_action_probs, qf1_next_target.logits, qf2_next_target.logits, next_state_log_pi
                )
            ]
            V_soft = sum(
                [
                    scatter(i, b, dim=0, dim_size=len(batch_idx), reduce="sum").sum(1)
                    for i, b in zip(min_qf_next_target, policy.batch)
                ]
            ).detach()
            Q_hat = self.gamma * torch.cat([V_soft[1:], torch.zeros_like(V_soft[:1])])
            Q_hat[final_graph_idx] = rewards + (1 - batch.is_valid) * self.invalid_penalty

        qf1_next, _ = critic1(batch, cond_info[batch_idx])
        qf2_next, _ = critic2(batch, cond_info[batch_idx])
        Q_sa1 = qf1_next.log_prob(batch.actions, logprobs=qf1_next.logits)
        Q_sa2 = qf2_next.log_prob(batch.actions, logprobs=qf2_next.logits)
        qf1_loss = F.mse_loss(Q_sa1, Q_hat).mean()
        qf2_loss = F.mse_loss(Q_sa2, Q_hat).mean()
        critic_loss = qf1_loss + qf2_loss

        # Actor training
        critic_loss.backward()
        opt_critic1.step()
        opt_critic1.zero_grad()
        opt_critic2.step()
        opt_critic2.zero_grad()

        policy, _ = actor(batch, cond_info[batch_idx])
        next_state_log_pi = policy.logsoftmax()
        next_state_action_probs = [torch.exp(lp) for lp in next_state_log_pi]

        with torch.no_grad():
            qf1_next, _ = critic1(batch, cond_info[batch_idx])
            qf2_next, _ = critic2(batch, cond_info[batch_idx])

        min_qf_next = [
            w * (self.alpha * lp - torch.min(qf1, qf2))
            for w, qf1, qf2, lp in zip(next_state_action_probs, qf1_next.logits, qf2_next.logits, next_state_log_pi)
        ]

        actor_loss = sum(
            [
                scatter(i, b, dim=0, dim_size=len(batch_idx), reduce="sum").sum(1)
                for i, b in zip(min_qf_next, policy.batch)
            ]
        ).mean()

        # Final loss
        loss = actor_loss
        invalid_mask = 1 - batch.is_valid
        info = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_loss": loss.item(),
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "rewards": rewards.mean(),
        }

        if not torch.isfinite(loss).all():
            raise ValueError("loss is not finite")
        return loss, info
