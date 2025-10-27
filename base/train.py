import os
import socket
import logging
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Dict

import torch 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .argument import TrainArgument
from .trainer import Trainer
from .util import setup_logging, ensure_reproducibility, get_open_port

T = TypeVar("T", bound=TrainArgument)
S = TypeVar("S", bound=TrainArgument)

ModelFactory = Callable[[torch.device], nn.Module]
DataLoaderFactory = Callable[[torch.device], DataLoader[Dict[str, torch.Tensor]]]
TrainerFactory = Callable[[Trainer.Options], Trainer]

class TrainApp(Generic[T]):
    """
    Train App
    """

    @dataclass(kw_only=True)
    class Options(Generic[S]):
        """
        Options for Trainer
        """

        model_factory: ModelFactory[S]
        data_loader_factory: DataLoaderFactory[S]
        trainer_factory: TrainerFactory[S]

    def __init__(self, args: T,opt:Options[T]) -> None:
        # TODO 这里是否需要硬编码CUDA编号？
        os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

        self.options = opt
        self.args = args

        self._setup_distributed_env()

        ngpu = torch.cuda.device_count()
        assert ngpu > 0,"No GPU available"

        logging.info(f"Found %d GPUs",ngpu)
        self.ngpu = ngpu

        # logging
        args_dict = {k: v for k, v in args.as_dict().items() if v is not None}
        logging.info("arguments: %s",args_dict)

        # ddp node
        self.node_size = int(os.getenv("NODE_SIZE","1"))
        self.node_rank = int(os.getenv("NODE_RANK","0"))
        logging.info("Run on node %d/%d",self.node_rank+1, self.node_size)

    def _setup_distributed_env(self):
        """ set env for distributed training """

        # set master_port
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(get_open_port())

        # set master_addr
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        host = socket.gethostbyname(socket.gethostname())
        logging.info("PyTorch DDP: addr = %s(%s), port = %s",
                    os.environ["MASTER_ADDR"],
                    host,
                    os.environ["MASTER_PORT"])

    def start(self) -> None:
        """
        Start training.
        """

        mp.spawn(
            _start_process,
            args=(self,),
            nprocs=self.ngpu,
            join=True,
            daemon=False
        )

def start_train(args: T,opt: TrainApp.Options[T]):
    """
    Start training.
    
    Args:
        args(T): The command line arguments.
        opt(TrainApp.Options[T]): The options for Trainer.
    """

    app = TrainApp(args, opt)
    app.start()

def _start_process(local_rank: int, app: TrainApp[T]) -> None:
    args = app.args
    opt = app.options

    setup_logging(args.log_level, prefix = f"Process- {local_rank}")
    # TODO 进程号还是 GPU 号？
    logging.info("Start training in num: %d GPU",local_rank)

    ensure_reproducibility(args.seed)
    logging.info("use seed %d",args.seed)

    ngpu = torch.cuda.device_count()
    world_size = ngpu * app.node_size
    global_rank = local_rank + app.node_rank * ngpu
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank
    )

    # device
    device = torch.device("cuda:%d" % local_rank)
    torch.cuda.set_device(device)

    # model
    model = opt.model_factory(device, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model parameters: %.2fM",(n_params / 1.0e6))
    model.to(device)

    # data loader
    data_loader = opt.data_loader_factory(device, args)

    # trainer
    trainer_opt = Trainer.Options(
        model = model,
        num_epoch = args.num_epoch,
        data_loader = data_loader,
        device = device,
        learning_rate = args.lr,
        weight_decay = args.weight_decay,
        global_rank = global_rank,
        local_rank = local_rank,
        acc_batch = args.acc_batch,
        checkpoint_dir = args.checkpoint_dir,
        checkpoint_prefix = args.checkpoint_prefix,
        max_time_sec = args.max_time_sec,
        use_amp = not args.no_use_amp
    )

    trainer = opt.trainer_factory(trainer_opt, args)
    trainer.start()