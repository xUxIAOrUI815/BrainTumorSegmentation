from typing import List, Callable, Optional
from dataclasses import dataclass

import torch

@dataclass(kw_only=True)
class Options:
    """ 
    Options for data loader.
    """

    batch_size: int = 1
    roi_sizes: Optional[List[int]] = None
    image_included: bool = True
    label_included: bool = False
    distributed: bool = False
    augument: bool = False
    device: Optional[torch.device] = None 
    no_random_rotate: bool = False
    no_crop_random_center: bool = False
    no_crop_foreground: bool = False
    prediction_path_lookup: Optional[Callable[[str], str]] = None

Option = Callable[[Options], None]

def with_batch_size(size: int) -> Option:
    """
    Set batch size.
    
    Args:
        size (int): The batch size.
        
    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.batch_size = size

    return _

def with_roi_sizes(sizes: List[int]) -> Option:
    """
    Set ROI sizes.
    
    Args:
        sizes (List[int]): The ROI sizes.
        
    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.roi_sizes = sizes

    return _

def with_distributed(distributed: bool) -> Option:
    """
    Set distributed mode.
    
    Args:
        distributed (bool): Whether to use distributed mode.
        
    Returns:
        Option: A dataset option.
    """

    def _(opt: Options):
        opt.distributed = distributed

    return _

def with_image_included(included: bool) -> Option:
    """
    Set whether to include images.
    
    Args:
        included (bool): Whether to include images.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.image_included = included

    return _

def with_label_included(included: bool) -> Option:
    """
    Set whether to include labels.
    
    Args:
        included (bool): Whether to include labels.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.label_included = included

    return _

def with_augument(augument: bool) -> Option:
    """
    Set whether to augument inputs.
    
    Args:
        augument (bool): Whether to augument inputs.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.augument = augument

    return _

def with_device(device: torch.device) -> Option:
    """
    Set device.
    
    Args:
        device (torch.device): The device.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.device = device

    return _

def with_no_random_rotate(no_random_rotate: bool) -> Option:
    """
    Set whether to disable random rotation.
    
    Args:
        no_random_rotate (bool): Whether to disable random rotation.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.no_random_rotate = no_random_rotate

    return _


def with_no_crop_random_center(no_crop_random_center: bool) -> Option:
    """
    Set whether to disable random center crop.
    
    Args:
        no_crop_random_center (bool): Whether to disable random center crop.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.no_crop_random_center = no_crop_random_center
    
    return _

def with_no_crop_foreground(no_crop_foreground: bool) -> Option:
    """
    Set whether to disable crop foreground.
    
    Args:
        no_crop_foreground (bool): Whether to disable crop foreground.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.no_crop_foreground = no_crop_foreground

    return _

def with_prediction_path_lookup(lookup: Callable[[str], str]) -> Option:
    """
    Set prediction path lookup function.
    
    Args:
        lookup (Callable[[str], str]): The lookup function.
        
    Returns:
        Option: A dataset option.
    """
    def _(opt: Options):
        opt.prediction_path_lookup = lookup

    return _
