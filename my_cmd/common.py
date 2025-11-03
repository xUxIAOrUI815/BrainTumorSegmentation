from model.network import SwinUNETR
from model.swin.swin import Swin, PatchEmbedding

from .argument import ModelArgument

def create_model(args: ModelArgument) -> SwinUNETR:

    opt = SwinUNETR.Options(
        output_channels = args.output_channel,
        swin_optins = Swin.Options(
            window_size = args.window_size,
            stage_depths = args.swin_stage_depths,
            stage_num_heads = args.swin_stage_num_heads,
            block_mlp_ratio = args.mlp_ratio,
            embed_options = PatchEmbedding.Options(
                num_dim=args.num_dim,
                patch_size = args.patch_size,
                input_channel=args.input_channel,
                output_channel=args.hidden_channel,
            ),
            # TODO: 创新模块
            # embed_options = MultiScalePatchEmbedding.Options(
            #     num_dim=args.num_dim,
            #     patch_size=args.patch_size,
            #     input_channel=args.input_channel,
            #     output_channel=args.hidden_channel,
            #     kernel_sizes=(1, 3, 5),
            # ),
        ),
    )
    return SwinUNETR(opt)