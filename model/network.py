from dataclasses import dataclass

import torch 
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Act

from .helper.padding import unpad_to_align
from .helper.swin import Swin

class SwinUNETR(nn.Module):
    """
    The Swin Module.
    """
    @dataclass(kw_only=True)
    class Options:
        """
        Options to build the model.
        """
        swin_options: Swin.Options
        output_channels: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        sopt = opt.swin_options
        eopt = sopt.embed_options
        num_stage = len(sopt.stage_depths)

        self.swin = Swin(sopt)
        self.encorder = Encoder(
            Encoder.Options(
                in_channel = eopt.input_channel,
                out_channel = eopt.output_channel,
                num_dim = eopt.num_dim
            ),
        )
        self.decoder = Decoder(
            Decoder.Options(
                in_channel = eopt.output_channel,
                out_channel = opt.output_channels,
                num_dim = eopt.num_dim,
                )
            )
        self.hidden_encoders = nn.ModuleList(
            [
                Encoder(
                    Encoder.Options(
                        in_channel = eopt.output_channel * (2**i),
                        out_channel = eopt.output_channel * (2**i),
                        num_dim = eopt.num_dim,
                    )
                )
            ]
        )
        self.hidden_decoders = nn.ModuleList(
            [
                Decoder(
                    Decoder.Options(
                        in_channel = eopt.output_channel * (2** (i + 1)),
                        out_channel = eopt.output_channel * (2**i),
                        num_dim = eopt.num_dim,
                    )
                )
                for i in range(num_stage)
            ]
        )

        self.out = Conv["conv", eopt.num_dim](
            in_channels = eopt.output_channel,
            out_channels = opt.output_channel,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.swin(x)
        es = [e(h) for e, h in zip(self.hidden_encoders, hs)]

        ds = [es[-1]]
        for decoder, e in zip(reversed(self,hidden_decoders), reversed(es[:-1])):
            d = decoder(ds[-1], e)
            ds.append(d)

        e = self.encoder(x)
        d = self.decoder(ds[-1], e)
        logits = self.out(d)
        probs = torch.sigmoid(logits)

        return probs
    
class Encoder(nn.Module):
    """
    The Encoder Module.
    """

    @dataclass(kw_only=True)
    class Options:
        """
        Options to build the module.
        """
        num_dim: int
        in_channel: int
        out_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        
        self.conv1 = Conv["conv", opt.num_dim](
            in_channels = opt.in_channel,
            out_channels = opt.out_channel,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

        self.norm1 = Norm["instance", opt.num_dim](
            num_features = opt.out_channel
        )

        self.conv2 = Conv["conv", opt.num_dim](
            in_channels = opt.out_channel,
            out_channel = opt.out_channel,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

        self.norm2 = Norm["instance", opt.num_dim](
            num_features = opt.out_channel
        )

        self.act = Act["leakyrelu"](
            inplace = True,
            negative_slope = 0.01,
        )

        self.proj = (
            Conv["conv", opt.num_dim](
                in_channels = opt.in_channel,
                out_channels = opt.out_channel,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            )
            if opt.in_channel != opt.out_channel 
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += self.proj(residual)
        x = self.act(x)

        return x
    
class Decoder(nn.Module):
    """
    The Decoder Module.
    """

    @dataclass(kw_only=True)
    class Options:
        """
        Options to build the module.
        """
        num_dim: int
        in_channel: int
        out_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        self.convtrans = Conv["convtrans", opt.num_dim](
            in_channels = opt.in_channel,
            out_channels = opt.out_channel,
            kernel_size = 2,
            stride = 2,
            padding = 0,
        )

        self.proj = Encoder(
            Encoder.Options(
                in_channel = opt.out_channel * 2,
                out_channel = opt.out_channel,
                num_dim = opt.num_dim,
            )
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.convtrans(x)

        x = unpad_to_align(x, a)

        x = torch.cat([x, a], dim = 1)
        x = self.proj(x)
        return x