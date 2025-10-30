import torch
import torch.nn as nn
from einops import rearrange
from monai.network.layers.weight_init import trunc_normal_

class WindowAttention(nn.Module):
    """
    The WindowAttention module
    """

    @dataclass(kw_only=True)
    class Options:
        """
        Options to build the model
        """

        num_dim: int
        window_size: int
        num_channel: int
        num_head: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        assert opt.num_channel % opt.num_head == 0

        self.scale = (opt.num_channel // opt.num_head) ** -0.5
        self.qkv = nn.Linear(
            in_features = opt.num_channel,
            out_features = opt.num_channel * 3,
            bias = False
        )
        self.softmax = nn.Softmax(dim = -1)

        self.relative_position_index = relative_position_index(
            dims = [opt.window_size] * opt.num_dim
        )

        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                opt.num_head,
                (2 * opt.window_size - 1) ** opt.num_dim
            )
        )
        trunc_normal_(self.relative_position_bias, std = 0.02)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor],
    )-> torch.Tensor:
        """
        Forward
        """

        opt = self.options

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "n v (k h c) -> k n v h c", h = opt.num_head, k = 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        p = q @ k.transpose(-2, -1)
        p += self.relative_position_bias[:, self.relative_position_index]
        if mask is not None:
            p += mask.unsqueeze(1)

        a = self.softmax(p)
        x = a @ v
        x = rearrange(x, "n v h c -> n v (h c)", h = opt.num_head)
        return x
    
def relative_position_index(dims: List[int])-> torch.Tensor:
    """
    Calculate the relative position index
    """

    coords = [torch.arange(d) for d in dims]
    coords = torch.stack(torch.meshgrid(*coords, indexing = "ij"))

    coords = torch.flatten(coords, 1)

    relative_coords = coords[:, :, None] - coords[:, None, :]

    relative_coords = relative_coords.permute(1, 2, 0)

    for i, d in enumerate(dims):
        relative_coords[:, :, i] += d - 1

    s = 1

    for i, d in enumerate(reversed(dims), 1):
        relative_coords[:, :, -i] *= s
        s *= 2 * d - 1

    index = relative_coords.sum(-1)
    return index