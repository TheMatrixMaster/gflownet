import math
import pathlib
import queue
import threading
from collections import defaultdict
from typing import List

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
from torch import Tensor

from gflownet.utils import metrics

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class MultiObjectiveStatsHook:
    """
    This hook is multithreaded and the keep_alive object needs to be closed for graceful termination.
    """

    def __init__(
        self,
        num_to_keep: int,
        log_dir: str,
        save_every: int = 50,
        compute_hvi=False,
        compute_hsri=False,
        compute_normed=False,
        compute_igd=False,
        compute_pc_entropy=False,
        compute_focus_accuracy=False,
        focus_cosim=None,
    ):
        # This __init__ is only called in the main process. This object is then (potentially) cloned
        # in pytorch data worker processed and __call__'ed from within those processes. This means
        # each process will compute its own Pareto front, which we will accumulate in the main
        # process by pushing local fronts to self.pareto_queue.
        self.num_to_keep = num_to_keep
        self.hsri_epsilon = 0.3

        self.compute_hvi = compute_hvi
        self.compute_hsri = compute_hsri
        self.compute_normed = compute_normed
        self.compute_igd = compute_igd
        self.compute_pc_entropy = compute_pc_entropy
        self.compute_focus_accuracy = compute_focus_accuracy
        self.focus_cosim = focus_cosim

        self.all_flat_rewards: List[Tensor] = []
        self.all_focus_dirs: List[Tensor] = []
        self.all_smi: List[str] = []
        self.pareto_queue: mp.Queue = mp.Queue()
        self.pareto_front = None
        self.pareto_front_smi = None
        self.pareto_metrics = mp.Array("f", 4)

        self.stop = threading.Event()
        self.save_every = save_every
        self.log_path = pathlib.Path(log_dir) / "pareto.pt"
        self.pareto_thread = threading.Thread(target=self._run_pareto_accumulation, daemon=True)
        self.pareto_thread.start()

    def _hsri(self, x):
        assert x.ndim == 2, "x should have shape (num points, num objectives)"
        upper = np.zeros(x.shape[-1]) + self.hsri_epsilon
        lower = np.ones(x.shape[-1]) * -1 - self.hsri_epsilon
        hsr_indicator = metrics.HSR_Calculator(lower, upper)
        try:
            hsri, _ = hsr_indicator.calculate_hsr(-x)
        except Exception:
            hsri = 1e-42
        return hsri

    def _run_pareto_accumulation(self):
        num_updates = 0
        timeouts = 0
        while not self.stop.is_set() and timeouts < 200:
            try:
                r, smi, owid = self.pareto_queue.get(block=True, timeout=1)
            except queue.Empty:
                timeouts += 1
                continue
            except ConnectionError as e:
                print("Pareto Accumulation thread Queue ConnectionError", e)
                break

            timeouts = 0
            # accumulates pareto fronts across batches
            if self.pareto_front is None:
                p = self.pareto_front = r
                psmi = smi
            else:
                p = np.concatenate([self.pareto_front, r], 0)
                psmi = self.pareto_front_smi + smi

            # distills down by removing dominated points
            idcs = metrics.is_pareto_efficient(-p, False)
            self.pareto_front = p[idcs]
            self.pareto_front_smi = [psmi[i] for i in idcs]

            # computes pareto metrics and store in multiprocessing array
            if self.compute_hvi:
                self.pareto_metrics[0] = metrics.get_hypervolume(torch.tensor(self.pareto_front), zero_ref=True)
            if self.compute_hsri:
                self.pareto_metrics[1] = self._hsri(self.pareto_front)
            if self.compute_igd:
                self.pareto_metrics[2] = metrics.get_IGD(torch.tensor(self.pareto_front))
            if self.compute_pc_entropy:
                self.pareto_metrics[3] = metrics.get_PC_entropy(torch.tensor(self.pareto_front))

            # saves data to disk
            num_updates += 1
            if num_updates % self.save_every == 0:
                if self.pareto_queue.qsize() > 10:
                    print("Warning: pareto metrics computation lagging")
                self._save()
        self._save()

    def _save(self):
        with open(self.log_path, "wb") as fd:
            torch.save(
                {
                    "pareto_front": self.pareto_front,
                    "pareto_metrics": list(self.pareto_metrics),
                    "pareto_front_smi": self.pareto_front_smi,
                },
                fd,
            )

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        # locally (in-process) accumulate flat rewards to build a better pareto estimate
        self.all_flat_rewards = self.all_flat_rewards + list(flat_rewards)
        if self.compute_focus_accuracy:
            self.all_focus_dirs = self.all_focus_dirs + list(cond_info["focus_dir"])
        self.all_smi = self.all_smi + list([i.get("smi", None) for i in trajs])
        if len(self.all_flat_rewards) > self.num_to_keep:
            self.all_flat_rewards = self.all_flat_rewards[-self.num_to_keep :]
            self.all_focus_dirs = self.all_focus_dirs[-self.num_to_keep :]
            self.all_smi = self.all_smi[-self.num_to_keep :]

        flat_rewards = torch.stack(self.all_flat_rewards).numpy()
        if self.compute_focus_accuracy:
            focus_dirs = torch.stack(self.all_focus_dirs).numpy()

        # collects empirical pareto front from in-process samples
        pareto_idces = metrics.is_pareto_efficient(-flat_rewards, return_mask=False)
        gfn_pareto = flat_rewards[pareto_idces]
        pareto_smi = [self.all_smi[i] for i in pareto_idces]

        # send pareto front to main process for lifetime accumulation
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        self.pareto_queue.put((gfn_pareto, pareto_smi, wid))

        # compute in-process pareto metrics and collects lifetime pareto metrics from main process
        info = {}
        if self.compute_hvi:
            unnorm_hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=True)
            unnorm_hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=False)
            info = {
                **info,
                "UHV, zero_ref=True": unnorm_hypervolume_with_zero_ref,
                "UHV, zero_ref=False": unnorm_hypervolume_wo_zero_ref,
                "lifetime_hv0": self.pareto_metrics[0],
            }
        if self.compute_normed:
            target_min = flat_rewards.min(0).copy()
            target_range = flat_rewards.max(0).copy() - target_min
            hypercube_transform = metrics.Normalizer(loc=target_min, scale=target_range)
            normed_gfn_pareto = hypercube_transform(gfn_pareto)
            hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=True)
            hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=False)
            info = {
                **info,
                "HV, zero_ref=True": hypervolume_with_zero_ref,
                "HV, zero_ref=False": hypervolume_wo_zero_ref,
            }
        if self.compute_hsri:
            hsri_w_pareto = self._hsri(gfn_pareto)
            info = {
                **info,
                "hsri": hsri_w_pareto,
                "lifetime_hsri": self.pareto_metrics[1],
            }
        if self.compute_igd:
            igd = metrics.get_IGD(flat_rewards, ref_front=None)
            info = {
                **info,
                "igd": igd,
                "lifetime_igd_frontOnly": self.pareto_metrics[2],
            }
        if self.compute_pc_entropy:
            pc_ent = metrics.get_PC_entropy(flat_rewards, ref_front=None)
            info = {
                **info,
                "PCent": pc_ent,
                "lifetime_PCent_frontOnly": self.pareto_metrics[3],
            }
        if self.compute_focus_accuracy:
            focus_acc = metrics.get_focus_accuracy(
                torch.tensor(flat_rewards), torch.tensor(focus_dirs), self.focus_cosim
            )
            info = {
                **info,
                "focus_acc": focus_acc,
            }

        return info

    def terminate(self):
        self.stop.set()
        self.pareto_thread.join()


class TopKHook:
    def __init__(self, k, repeats, num_preferences):
        self.queue: mp.Queue = mp.Queue()
        self.k = k
        self.repeats = repeats
        self.num_preferences = num_preferences

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        self.queue.put([(i["data_idx"], r) for i, r in zip(trajs, rewards)])
        return {}

    def finalize(self):
        data = []
        while not self.queue.empty():
            try:
                data += self.queue.get(True, 1)
            except queue.Empty:
                # print("Warning, TopKHook queue timed out!")
                break
        repeats = defaultdict(list)
        for idx, r in data:
            repeats[idx // self.repeats].append(r)
        top_ks = [np.mean(sorted(i)[-self.k :]) for i in repeats.values()]
        assert len(top_ks) == self.num_preferences  # Make sure we got all of them?
        return top_ks


class RewardPercentilesHook:
    """
    Calculate percentiles of the reward.

    Parameters
    ----------
    idx: List[float]
        The percentiles to calculate. Should be in the range [0, 1].
        Default: [1.0, 0.75, 0.5, 0.25, 0]
    """

    def __init__(self, percentiles=None):
        if percentiles is None:
            percentiles = [1.0, 0.75, 0.5, 0.25, 0]
        self.percentiles = percentiles

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        x = np.sort(flat_rewards.numpy(), axis=0)
        ret = {}
        y = np.sort(rewards.numpy())
        for p in self.percentiles:
            f = max(min(math.floor(x.shape[0] * p), x.shape[0] - 1), 0)
            for j in range(x.shape[1]):
                ret[f"percentile_flat_reward_{j}_{p:.2f}"] = x[f, j]
            ret[f"percentile_reward_{p:.2f}%"] = y[f]
        return ret


class TrajectoryLengthHook:
    """
    Report the average trajectory length along with number of atoms and bonds
    """

    def __init__(self) -> None:
        pass

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        ret = {}
        ret["avg_traj_length"] = sum([len(i["traj"]) for i in trajs]) / len(trajs)
        ret["max_traj_length"] = max([len(i["traj"]) for i in trajs])
        return ret


class AtomicPropertiesHook:
    """
    Report the average atomic properties along with number of atoms and bonds
    """

    def __init__(self) -> None:
        pass

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        ret = {}
        mols = [i["mol"] for i in trajs if i["mol"] is not None]
        n_atoms = torch.tensor([m.GetNumAtoms() for m in mols], dtype=torch.float32)
        n_bonds = torch.tensor([m.GetNumBonds() for m in mols], dtype=torch.float32)
        ret["avg_number_atoms"] = n_atoms.mean().item()
        ret["avg_number_bonds"] = n_bonds.mean().item()
        return ret


class SnapshotDistributionHook:
    """
    Keeps a list of all the molecules sampled along with their rewards and
    produces either a reward distribution plot or a tanimoto similarity plot

    TODO: Support multiple workers by using a multiprocessing.Queue
    """

    def __init__(self) -> None:
        self.mols = []
        self.fpgen = AllChem.GetRDKitFPGenerator()

    def plot_reward_distribution(self):
        rewards = [i[1] for i in self.mols]
        fig, ax = plt.subplots()
        ax.hist(rewards, bins=50)
        ax.set_xlabel("Reward")
        ax.set_ylabel("Density")
        return fig

    def plot_top_k_tanimoto_similarity(self, k=64):
        k = min(k, len(self.mols))
        mols_sorted = sorted(self.mols, key=lambda x: x[1], reverse=True)
        top_k_mols = [i[0] for i in mols_sorted[:k]]
        top_k_fps = [self.fpgen.GetFingerprint(mol) for mol in top_k_mols]

        tanimoto_sim = []
        for i, j in combinations(top_k_fps, 2):
            tanimoto_sim.append(AllChem.DataStructs.TanimotoSimilarity(i, j))

        fig, ax = plt.subplots()
        ax.hist(tanimoto_sim, bins=50)
        ax.set_title(f"Top-{k} Tanimoto Similarity")
        ax.set_xlabel("Tanimoto Similarity")
        ax.set_ylabel("Density")
        return fig

    def plot_tanimoto_similarity_to_target(self, target_mol):
        target_fp = self.fpgen.GetFingerprint(target_mol)
        fps = [self.fpgen.GetFingerprint(i[0]) for i in self.mols]
        tanimoto_sim = AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, fps)

        fig, ax = plt.subplots()
        ax.hist(tanimoto_sim, bins=50)
        ax.set_title("Tanimoto Similarity to Target")
        ax.set_xlabel("Tanimoto Similarity")
        ax.set_ylabel("Density")
        return fig

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        for traj, reward in zip(trajs, rewards):
            mol = traj["mol"]
            if mol is not None:
                self.mols.append((mol, reward))
        return {}


class NumberOfScaffoldsHook:
    """Plots number of unique molecular scaffolds throughout training"""
    def __init__(self) -> None:
        self.scaffolds = set()

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        for traj in trajs:
            mol = traj["mol"]
            if mol is not None:
                scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
                self.scaffolds.add(scaffold)
        return {"num_scaffolds": len(self.scaffolds)}


class NumberOfModesHook:
    """
    Keeps a list of the "modes" sampled along with their reward. A mode is defined as a
    molecule whose Tanimoto similarity to any other molecule in the list is below a
    certain threshold.

    TODO: Support multiple workers by using a multiprocessing.Queue
    """

    reward_mode: str                # "hard" or "percentile"
    sim_thresholds: List[float]     # similarity thresholds to log
    reward_threshold: float         # reward threshold to log
    stop_logging_after: int         # number of modes after which to stop logging

    def __init__(
        self,
        reward_mode="hard",
        reward_threshold=0.9,
        sim_thresholds=[0.7, 0.5, 0.2],
    ) -> None:
        self.sim_thresholds = sim_thresholds
        self.sim_high = max(sim_thresholds)
        self.reward_mode = reward_mode
        self.reward_threshold = reward_threshold
        self.modes = defaultdict(list)
        self.mode_fps = defaultdict(list)
        self.stop_logging_after = 2000
        self.fpgen = AllChem.GetRDKitFPGenerator()

        assert self.reward_mode in ["hard", "percentile"]
        if self.reward_mode == "percentile":
            raise NotImplementedError

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        res = {}
        for traj, reward in zip(trajs, rewards):
            mol = traj["mol"]
            if mol is None or reward < self.reward_threshold: continue
            fp = self.fpgen.GetFingerprint(mol)

            for sim_thresh in self.sim_thresholds:
                if self.should_stop_logging(sim_thresh):
                    continue
                simkey = self.get_key_from_sim(sim_thresh)
                if metrics.all_are_tanimoto_different(sim_thresh, fp, self.mode_fps[simkey]):
                    self.modes[simkey].append((mol, reward))
                    self.mode_fps[simkey].append(fp)

                res[f"{self.__label__(sim_thresh)}"] = len(self.modes[simkey])
        return res

    def should_stop_logging(self, sim_thresh: float):
        simkey = self.get_key_from_sim(sim_thresh)
        if self.stop_logging_after is None or simkey not in self.modes: return False
        return len(self.modes[simkey]) >= self.stop_logging_after

    def split_by_scaffold(self):
        """Splits the modes by scaffold then sorts by descending reward"""
        simkey = self.get_key_from_sim(self.sim_high)
        scaffold_to_modes = defaultdict(list)
        for mol, reward in self.modes[simkey]:
            scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
            scaffold_to_modes[scaffold].append((mol, reward))
        scaffold_to_modes = {
            scaffold: sorted(modes, key=lambda x: x[1], reverse=True) for scaffold, modes in scaffold_to_modes.items()
        }
        return scaffold_to_modes

    def get_key_from_sim(self, sim_thresh: float):
        assert sim_thresh != None
        return f'sim<={sim_thresh}'
    
    def __label__(self, sim_thresh: float):
        return f"modes_>=_{self.reward_threshold:.2f}_sim_<=_{sim_thresh:.2f}"


class NumberOfUniqueTrajectoriesHook:
    """Counts the number of unique trajectories sampled over training"""

    def __init__(self) -> None:
        self.unique_trajs = set()

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        for traj in trajs:
            mol = traj["mol"]
            if mol is not None:
                self.unique_trajs.add(Chem.MolToSmiles(mol))
        return {"num_unique_trajs": len(self.unique_trajs)}
