import os
import random
import logging
import socket
from contextlib import closing

import numpy 
import torch

def ensure_reproducibility(seed: int) -> None:
    """
    Ensure reproducibility.

    Args:
        seed (int): The seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(level: int, prefix: str = ""):
    """
    Setup logging.
    
    Args:
        level (int): The log level.
        prefix (str, optional): The log prefix. Defaults to "".
    """
    numeric_level = getattr(logging, level.upper(), None)
    assert isinstance(numeric_level, int)

    components = ["%(asctime)s.%(msecs)03d","%(levelname)s"]
    if prefix != "":
        components.append(prefix)
    components.append("%(message)s")

    logging.basicConfig(
        level = numeric_level,
        datefmt = "%Y-%m-%d %H:%M:%S",
        format = " ".join(components),
        force = True,
    )


def get_open_port() :
    """
    Get an open port.

    Returns:
        int: The open port.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]




