import abc
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .trainer import CKPT_MODEL_KEY, TensorDict

class Predictor:
    """
    Class for running model ptrdictions.
    """

    @dataclass(kw_only=True)
    class Options:
        """
        Options for Predictor
        """

        model: nn.Module
        data_loader: DataLoader[TensorDict]
        device: torch.device
        checkpoint: str
        output_dir: str

    def __init__(self, opt: Options) -> None:
        self.options = opt
        self._load_checkpoint()

    @abc.abstractmethod
    def forward(self, inferer: nn.Module, batch: TensorDict) -> None:
        """
        Forward and compute loss.

        Args:
            inferer (nn.Module): Preform forward computation.
            batch (TensorDict): Input batch.
        """

        raise NotImplementedError
    
    def start(self) :
        """
        Start predicting.
        """

        opt = self.options

        logging.info("Start inference")
        opt.model.eval()

        with torch.no_grad():
            for bidx, batch in enumerate(opt.data_loader, 1):
                start_time = time.time()

                self.forward(opt.model, batch)
                logging.info(
                    "Finish batch %d/%d with duration %.2fs",
                    bidx,
                    len(opt.data_loader),
                    time.time() - start_time,
                )

    def _load_checkpoint(self) -> None:
        """
        Load checkpoint.
        """

        opt = self.options

        logging.info("Load checkpoint from %s", opt.checkpoint)
        checkpoint = torch.load(opt.checkpoint, map_location=opt.device)
        opt.model.load_state_dict(checkpoint[CKPT_MODEL_KEY])
        opt.model.to(opt.device) 
        logging.info("Checkpoint loaded")