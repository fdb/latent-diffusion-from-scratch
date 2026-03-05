#!/bin/bash
# Launch paired conditional diffusion training.
# Auto-detects available GPUs and uses all of them via Accelerate.
#
# Usage:
#   ./start_training.sh                          # train from scratch
#   ./start_training.sh --resume_from output/... # resume from checkpoint
#   ./start_training.sh --batch_size 8 --learning_rate 2e-5
#
# On a single GPU (e.g. RTX 5090), this runs normal training.
# On a multi-GPU node (e.g. 8x H100 SXM), each GPU gets a data shard
# and gradients are synced automatically via DDP.
#
# Note: effective batch size = batch_size * num_gpus * gradient_accumulation_steps.
# You may want to scale --learning_rate linearly with the number of GPUs.

set -e

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "Detected $NUM_GPUS GPU(s)"

accelerate launch --num_processes="$NUM_GPUS" train_paired_256.py "$@"
