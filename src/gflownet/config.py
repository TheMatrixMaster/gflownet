from dataclasses import dataclass, field, fields, asdict, is_dataclass, Field
from typing import Optional, List, Any
from copy import copy

from omegaconf import MISSING

from gflownet.algo.config import AlgoConfig, TBVariant
from gflownet.data.config import ReplayConfig
from gflownet.models.config import ModelConfig, SeqPosEnc
from gflownet.tasks.config import TasksConfig
from gflownet.utils.config import ConditionalsConfig


@dataclass
class OptimizerConfig:
    """Generic configuration for optimizers

    Attributes
    ----------
    opt : str
        The optimizer to use (either "adam" or "sgd")
    learning_rate : float
        The learning rate
    lr_decay : float
        The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))
    weight_decay : float
        The L2 weight decay
    momentum : float
        The momentum parameter value
    clip_grad_type : str
        The type of gradient clipping to use (either "norm" or "value")
    clip_grad_param : float
        The parameter for gradient clipping
    adam_eps : float
        The epsilon parameter for Adam
    """

    opt: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 20_000
    weight_decay: float = 1e-8
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8


@dataclass
class Config:
    """Base configuration for training

    Attributes
    ----------
    desc : str
        A description of the experiment
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda[:<device_id>]")
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    store_all_checkpoints : bool
        Whether to store all checkpoints or only the last one
    print_every : int
        The number of training steps after which to print the training loss
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    desc: str = "noDesc"
    log_dir: str = MISSING
    device: str = "cuda"
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: Optional[int] = None
    store_all_checkpoints: bool = False
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: Optional[int] = None
    num_validation_gen_steps: Optional[int] = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: Optional[str] = None
    pickle_mp_messages: bool = False
    git_hash: Optional[str] = None
    overwrite_existing_exp: bool = False
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    task: TasksConfig = field(default_factory=TasksConfig)
    cond: ConditionalsConfig = field(default_factory=ConditionalsConfig)


def init_empty(cfg):
    """
    Initialize a dataclass instance with all fields set to MISSING,
    including nested dataclasses.

    This is meant to be used on the user side (tasks) to provide
    some configuration using the Config class while overwritting
    only the fields that have been set by the user.
    """
    for f in fields(cfg):
        if is_dataclass(f.type):
            setattr(cfg, f.name, init_empty(f.type()))
        else:
            setattr(cfg, f.name, MISSING)

    return cfg


def init_from_dict(cfg: Config, d: dict) -> Config:
    """
    Initialize a dataclass instance with all fields set to MISSING,
    except for the fields that are present in the provided dictionary.
    """
    for f in fields(cfg):
        if is_dataclass(f.type):
            r = init_from_dict(f.type(), d[f.name])
            setattr(cfg, f.name, r)
        elif f.name in d:
            setattr(cfg, f.name, d[f.name])
        else:
            setattr(cfg, f.name, MISSING)

    return cfg


def merge_cfgs(base: Config, override: Config) -> Config:
    """
    Merge two config objects such that the fields of the override
    object take precedence over the base object. This is meant to
    be used to merge the user provided configuration with the default
    configuration.
    """
    for f in fields(override):
        v = getattr(override, f.name)
        if is_dataclass(f.type):
            new_v = merge_cfgs(getattr(base, f.name), v)
        elif v != MISSING:
            continue
        else:
            new_v = getattr(base, f.name)
        setattr(override, f.name, new_v)
    return override


def mix_and_mash(cfgs: List[Config], sub_cfgs: List[Any], f: Field) -> List[Config]:
    """
    Compute the cartesian product of parent and child configs
    """
    new_cfgs = []
    for cfg in cfgs:
        for sub_cfg in sub_cfgs:
            setattr(cfg, f.name, sub_cfg)
            new_cfgs.append(copy(cfg))
    return new_cfgs


def dfs_config_tree(tree) -> List[Config]:
    """
    Depth-first search to create all config combinations from a config tree
    """
    cfgs = [init_empty(copy(tree))]
    for f in fields(tree):
        v = getattr(tree, f.name)
        if v == MISSING:
            continue
        elif is_dataclass(f.type):
            sub_cfgs = dfs_config_tree(v)
        else:
            assert len(v) > 0, "Each field in the config tree must be a list of values"
            sub_cfgs = v
        cfgs = mix_and_mash(cfgs, sub_cfgs, f)
    return cfgs


def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, TBVariant):
            return obj.value
        elif isinstance(obj, SeqPosEnc):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data if v != MISSING)


def strip_missing(cfg: Config) -> dict:
    """
    Recursively remove all fields in cfg that value MISSING
    """
    return asdict(cfg, dict_factory=custom_asdict_factory)
