"""Export paired conditional diffusion UNet to ONNX format.

Loads a training checkpoint and exports the 6-channel UNet (noisy target + source
conditioning) to ONNX for inference on Mac via ONNX Runtime.

Usage:
    python export_unet_onnx.py --checkpoint_dir output/.../checkpoints/checkpoint-0004
    python export_unet_onnx.py --checkpoint_dir output/.../checkpoints/checkpoint-0004 --fp16
"""

import argparse
import os

import torch
from diffusers import UNet2DModel


class UNetWrapper(torch.nn.Module):
    """Thin wrapper to make UNet2DModel export-friendly for ONNX.

    UNet2DModel returns a dataclass with .sample; this wrapper returns the
    raw noise prediction tensor directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep):
        return self.model(sample, timestep).sample


def main():
    parser = argparse.ArgumentParser(description="Export UNet to ONNX")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint dir containing config.json and unet_state_dict.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx path (defaults to checkpoint_dir/unet.onnx)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Also export an fp16 version for smaller/faster inference on Apple Silicon",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.checkpoint_dir, "unet.onnx")

    # Load model architecture from saved config
    model = UNet2DModel.from_config(args.checkpoint_dir)

    # Load trained weights
    state_dict_path = os.path.join(args.checkpoint_dir, "unet_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded UNet from {args.checkpoint_dir}")
    print(f"  in_channels={model.config.in_channels}, out_channels={model.config.out_channels}")

    wrapper = UNetWrapper(model)

    # Dummy inputs matching training: 6-channel input at 256x256, scalar timestep per batch item
    dummy_sample = torch.randn(1, 6, 256, 256)
    dummy_timestep = torch.tensor([999], dtype=torch.long)

    print(f"Exporting to {output_path} (opset {args.opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_sample, dummy_timestep),
        output_path,
        export_params=True,
        opset_version=args.opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=["sample", "timestep"],
        output_names=["noise_pred"],
        dynamic_axes={
            "sample": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "noise_pred": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported fp32 ONNX model: {output_path} ({file_size_mb:.1f} MB)")

    if args.fp16:
        from onnxconverter_common import float16
        import onnx

        fp16_path = output_path.replace(".onnx", "_fp16.onnx")
        print(f"Converting to fp16: {fp16_path}...")

        model_fp32 = onnx.load(output_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, fp16_path)

        fp16_size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
        print(f"Exported fp16 ONNX model: {fp16_path} ({fp16_size_mb:.1f} MB)")
        print(f"Size reduction: {file_size_mb:.1f} MB -> {fp16_size_mb:.1f} MB ({fp16_size_mb/file_size_mb*100:.0f}%)")


if __name__ == "__main__":
    main()
