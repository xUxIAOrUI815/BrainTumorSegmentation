import os
from typing import List, Dict, Any, cast
from dataclasses import dataclass

import torch
from torch import nn

import nibabel as nib
from monai.inferers.inferer import SlidingWindowInferer

from base.predictor import Predictor as BasePredictor
from dataset.loader import IMAGE_KEY

class Predictor(BasePredictor):
    """ Class for running a model. """

    @dataclass(kw_only = True)
    class Options(BasePredictor.Options):
        """ Options for Predictor. """
        sw_roi_sizes: List[int]     # 滑动窗口的感兴趣区域大小
        sw_overlap_ratio: float     # 窗口之间的重叠比例
        sw_batch_size: int = 1      # 滑动窗口的批量大小

    def __init__(self, opt: Options) -> None:
        super().__init__(opt)

        self.options = opt
        self.inferer = SlidingWindowInferer(
            roi_size = opt.sw_roi_sizes,        # 每个窗口的大小
            sw_batch_size = opt.sw_batch_size,  # 并行处理的窗口数
            overlap = opt.sw_overlap_ratio,     # 窗口重叠度
        )

    def forward(self, inferer: nn.Module, batch: Dict[str, Any]) -> None:
        opt = self.options

        image = batch[IMAGE_KEY]
        image = image.to(opt.device)

        probs = self.inferer(inputs = image, network = inferer)

        probs = cast(torch.Tensor, probs)

        pred = convert_from_multi_channel(probs)
        pred = pred.detach().cpu().numpy()
        metadata = batch[f"{IMAGE_KEY}_meta_dict"]
        for bidx, fpath in enumerate(metadata["filename_or_obj"]):
            case_name = os.path.basename(os.path.dirname(fpath))
            save_dir = os.path.join(opt.output_dir, case_name)
            sava_path = os.path.join(save_dir, f"{case_name}_pred.nii.gz")
            os.makedirs(save_dir, exist_ok = True)

            affine = metadata["affine"][bidx]
            nifti = nib.Nifti1Image(pred[bidx], affine)
            nib.save(nifti, sava_path)

def convert_from_multi_channel(probs: torch.Tensor) -> torch.Tensor:
    # 将多通道 one-hot 概率图转为单通道标签图
    """ Convert multi-channel probabilities to single-channel. """

    # 概率二值化
    seg = (probs > 0.5).to(torch.uint8)

    [batch_size, _, *dims] = seg.shape
    seg_out = torch.zeros([batch_size, *dims])

    # 多通道到单通道的映射
    seg_out[seg[:, 1] == 1] = 2
    seg_out[seg[:, 0] == 1] = 1
    seg_out[seg[:, 2] == 1] = 4

    seg_out = seg_out.to(torch.uint8)

    return seg_out