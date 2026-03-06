#!/bin/sh
mkdir -p output/models
curl -o output/models/vae_decoder.onnx https://algorithmicgaze.s3.amazonaws.com/projects/2026-latent-diffusion-from-scratch/models/vae_decoder.onnx
