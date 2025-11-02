import torch
from torch import nn

from monai.losses.dice import DiceLoss

from base.trainer import Trainer as BaseTrainer, TensorDict
from dataset.loader import IMAGE_KEY, LABEL_KEY

class Trainer(BaseTrainer):
    """
    Class for training a model
    """

    def __init__(self, opt: BaseTrainer.Options) -> None:
        super().__init__(opt)

        self.loss_func = DiceLoss()

    def forward_loss(self, inferer: nn.Module, batch: TensorDict) -> torch.Tensor:
        opt = self.options
        
        image = batch[IMAGE_KEY]
        label = batch[LABEL_KEY]

        probs = inferer(image)
        loss = self.loss_func(probs, label)
        return loss

# TODO 确认是否需要进行数据转换，检查输入和输出数据的格式和通道数、标签数、编码方式，是否需要转成one-hot编码
# TODO 尝试其他损失函数