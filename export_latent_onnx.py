"""Export latent paired diffusion models to ONNX format.

Exports three ONNX models:
  1. vae_encoder.onnx — encodes 256x256 RGB to 32x32x4 latent (with scaling factor baked in)
  2. unet.onnx — 8-channel latent UNet (4 noisy + 4 conditioning)
  3. vae_decoder.onnx — decodes 32x32x4 latent back to 256x256 RGB (with scaling factor baked in)

Usage:
    uv run python export_latent_onnx.py --checkpoint_dir output/.../checkpoints/checkpoint-0004
    uv run python export_latent_onnx.py --checkpoint_dir output/.../checkpoints/checkpoint-0004 --fp16
"""

import argparse
import os

import torch
from diffusers import AutoencoderKL, UNet2DModel

VAE_MODEL_ID = "stabilityai/sd-vae-ft-msa"
VAE_SCALING_FACTOR = 0.18215


class VAEEncoderWrapper(torch.nn.Module):
    """Wraps VAE encoder for ONNX export, bypassing DiagonalGaussianDistribution."""

    def __init__(self, vae, scaling_factor=VAE_SCALING_FACTOR):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.scaling_factor = scaling_factor

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * self.scaling_factor


class VAEDecoderWrapper(torch.nn.Module):
    """Wraps VAE decoder for ONNX export."""

    def __init__(self, vae, scaling_factor=VAE_SCALING_FACTOR):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.scaling_factor = scaling_factor

    def forward(self, z):
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)


class UNetWrapper(torch.nn.Module):
    """Wraps UNet2DModel for ONNX export, returning raw tensor instead of dataclass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep):
        return self.model(sample, timestep).sample


def export_model(wrapper, dummy_inputs, output_path, input_names, output_names,
                 dynamic_axes, opset_version):
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported: {output_path} ({file_size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export latent diffusion models to ONNX")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to checkpoint dir containing config.json and unet_state_dict.pt",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for ONNX files (defaults to checkpoint_dir)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Also export fp16 versions for smaller/faster inference",
    )
    parser.add_argument(
        "--opset_version", type=int, default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint_dir

    # --- Load VAE ---
    print(f"Loading VAE from {VAE_MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID)
    vae.eval()

    # --- Load UNet ---
    print(f"Loading UNet from {args.checkpoint_dir}...")
    unet = UNet2DModel.from_config(args.checkpoint_dir)
    state_dict_path = os.path.join(args.checkpoint_dir, "unet_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    unet.load_state_dict(state_dict)
    unet.eval()
    print(f"  in_channels={unet.config.in_channels}, out_channels={unet.config.out_channels}")

    exported_paths = []

    # --- Export VAE Encoder ---
    print("Exporting VAE encoder...")
    encoder_wrapper = VAEEncoderWrapper(vae)
    encoder_wrapper.eval()
    encoder_path = export_model(
        encoder_wrapper,
        (torch.randn(1, 3, 256, 256),),
        os.path.join(output_dir, "vae_encoder.onnx"),
        input_names=["x"],
        output_names=["latent"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "latent": {0: "batch_size"},
        },
        opset_version=args.opset_version,
    )
    exported_paths.append(encoder_path)

    # --- Export UNet ---
    print("Exporting UNet...")
    unet_wrapper = UNetWrapper(unet)
    unet_wrapper.eval()
    unet_path = export_model(
        unet_wrapper,
        (torch.randn(1, 8, 32, 32), torch.tensor([999], dtype=torch.long)),
        os.path.join(output_dir, "unet.onnx"),
        input_names=["sample", "timestep"],
        output_names=["noise_pred"],
        dynamic_axes={
            "sample": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "noise_pred": {0: "batch_size"},
        },
        opset_version=args.opset_version,
    )
    exported_paths.append(unet_path)

    # --- Export VAE Decoder ---
    print("Exporting VAE decoder...")
    decoder_wrapper = VAEDecoderWrapper(vae)
    decoder_wrapper.eval()
    decoder_path = export_model(
        decoder_wrapper,
        (torch.randn(1, 4, 32, 32),),
        os.path.join(output_dir, "vae_decoder.onnx"),
        input_names=["z"],
        output_names=["decoded"],
        dynamic_axes={
            "z": {0: "batch_size"},
            "decoded": {0: "batch_size"},
        },
        opset_version=args.opset_version,
    )
    exported_paths.append(decoder_path)

    # --- Optional fp16 conversion ---
    if args.fp16:
        from onnxconverter_common import float16
        import onnx

        for path in exported_paths:
            fp16_path = path.replace(".onnx", "_fp16.onnx")
            print(f"Converting to fp16: {fp16_path}...")
            model_fp32 = onnx.load(path)
            model_fp16 = float16.convert_float_to_float16(model_fp32)
            onnx.save(model_fp16, fp16_path)
            fp16_size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
            fp32_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {fp32_size_mb:.1f} MB -> {fp16_size_mb:.1f} MB ({fp16_size_mb/fp32_size_mb*100:.0f}%)")

    print("\nDone! Exported files:")
    for path in exported_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
