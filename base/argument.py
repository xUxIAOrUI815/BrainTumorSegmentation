import os
import random
from typing import Optional, List, Any, Dict

from tap import Tap

class BaseArgument(Tap):
    """
    Base Argument class
    """

    def __init__(
            self,
            *args: List[Any],
            underscores_to_dashes: bool = False,
            explicit_bool: bool = False,
            config_files: Optional[List[str]] = None,
            **kwargs: Dict[str, Any],
    ):
        config_files = config_files or []
        if (env := os.getenv("ARGS_FILES")) is not None:
            extras = env.split(" ")
            config_files.extend(extras)

        super().__init__(
            *args,
            underscores_to_dashes=underscores_to_dashes,
            explicit_bool=explicit_bool,
            config_files=config_files,
            **kwargs,
        )

class CommonArgument(BaseArgument):
    """
    Common Argument class
    """

    log_level: str = "INFO"

    data_root: str

    fold_map: str 

    folds: List[int] 

    seed: int = random.randint(2**32)


class DatasetArgument(CommonArgument):
    """
    Dataset Argument class
    """

    batch_size: int = 1 # 批次大小默认为 1


class TrainArgument(CommonArgument):
    """
    Train Argument class
    """

    num_epoch: int

    max_time_sec: Optional[int] = None  

    acc_batch: int = 1      # 梯度累积步数

    lr: float = 1e-4

    weight_decay: float = 1e-5  # 权重衰减

    checking_dir: str       # 检查点目录

    checkpoint_prefix: str  # 检查点前缀

    no_use_amp: bool = False    # 默认启动混合精度训练


class PredictArgument(CommonArgument):
    """
    Predict Argument class
    """

    checkpoint_path: str

    output_dir: str