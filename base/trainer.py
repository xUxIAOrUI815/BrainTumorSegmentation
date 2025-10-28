import os
import abc
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data.dataloader import DataLoader


TensorDict = Dict[str, torch.Tensor]

CKPT_MODEL_KEY = "model"
CKPT_EPOCH_KEY = "epoch"
CKPT_OPTIMIZER_KEY = "optimizer"
CKPT_SCHEDULER_KEY = "scheduler"


class Trainer:
    """
    class for training a model
    """

    @dataclass(kw_only=True)
    class Options:
        """
        Options for Trainer
        """

        model: nn.Module
        data_loader:DataLoader[TensorDict]
        
        device: torch.device
        global_rank: int
        local_rank: int

        num_epoch: int
        learning_rate: float
        weight_decay: float
        acc_batch: int = 1
        use_amp: bool = True
        max_time_sec: Optional[int] = None

        checkpoint_dir: str
        checkpoint_prefix: str = "model"        # 保存路径的前缀

    def __init__(self, opt:Options) -> None:
        self.options = opt
        logging.info("Accumulate %d batch(es) before updating gradients",opt.acc_batch)

        self.optimizer = torch.optim.AdamW(
            opt.model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=opt.num_epoch,
        )

        # DDP
        self._ddp_model = torch.nn.parallel.DistributedDataParallel(
            opt.model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank
        )

    @abc.abstractmethod
    def forward_loss(self, inferer: nn.Module, batch: TensorDict) -> TensorDict:
        """
        forward and compute loss
        """

        raise NotImplementedError
    

    def start(self):
        """
        start training
        """

        opt = self.options

        logging.info("Start training")
        self._ddp_model.train()

        start_time = time.time()
        finished_epoch = 0
        for eidx in range(1, opt.num_epoch +1):
            opt.data_loader.sample.set_epoch(eidx - 1)
            dist.barrier()

            logging.info(
                "Start epoch %d/%d with learning rate: %s",
                eidx,
                opt.num_epoch,
                self.scheduler.get_last_lr(),
            )

            epoch_start_time = time.time()
            loss = self._train_epoch()
            logging.info(
                "Finished epoch %d/%d in %.2fs with loss: %.4f",
                eidx,
                opt.num_epoch,
                time.time() - epoch_start_time,
                loss,
            )

            finished_epoch += 1

            if (t_s := opt.max_time_sec) is not None:
                past = time.time() - start_time
                if past > t_s:
                    logging.info("Terminate training due to time limit (%ds)",t_s)
                    break

            self.scheduler.step()

        self._save_checkpoint(finished_epoch)
        logging.info("Finished training")

    
    def _train_epoch(self) -> float:
        """
        Train an epoch.

        Returns:
            float: The epoch loss.
        """

        opt = self.options

        total_loss = 0.0
        total_sample = 0.0 

        scaler = GradScaler() if opt.use_amp else None

        batch_size = opt.data_loader.batch_size
        assert batch_size is not None

        total_batch = len(opt.data_loader)
        for idx, batch in enumerate(opt.data_loader, 1):
            logging.info("Start batch %d/%d",idx, total_batch)

            # forward
            with autocast(enabled=opt.use_amp):
                loss = self.forward_loss(
                    inferer = self._ddp_model,
                    batch = batch,
                )
            loss /= opt.acc_batch

            # backward
            if opt.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (idx % opt.acc_batch == 0) or (idx == total_batch):
                logging.debug("Apply gradient update")
                if opt.use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # collect
            total_loss += loss.item() * opt.acc_batch
            total_sample += batch_size

        # TODO 检查不充分？
        if total_sample == 0:
            logging.warning("No sample is processed")
            return -1.0
        
        return total_loss / total_sample
    
    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save checkpoint.

        Args:
            epoch (int): The epoch number.
        """

        opt = self.options

        if opt.global_rank != 0:
            return

        state_dict = self.options.model.state_dict()
        save_name = f"{opt.checkpoint_prefix}-{epoch}.pt"
        save_path = os.path.join(opt.checkpoint_dir, save_name)
        torch.save(  # type: ignore
            {
                f"{CKPT_MODEL_KEY}": state_dict,
                f"{CKPT_EPOCH_KEY}": epoch,
                f"{CKPT_OPTIMIZER_KEY}": self.optimizer.state_dict(),  # type: ignore
                f"{CKPT_SCHEDULER_KEY}": self.scheduler.state_dict(),  # type: ignore
            },
            save_path,
        )
        logging.info("Saved checkpoint at %s", save_path)


# TODO 早停机制 


