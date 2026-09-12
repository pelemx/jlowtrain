import os
import torch

# ============================================================
# JLOW V2 — GPU 1, MAX 20 GiB VRAM
# ============================================================

# Physical GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    GPU_INDEX = torch.cuda.current_device()

    # Limit THIS PyTorch process to 20 GiB of GPU memory.
    # RTX 6000 = 24 GiB total.
    MAX_VRAM_GIB = 20
    TOTAL_VRAM_GIB = 24

    torch.cuda.set_per_process_memory_fraction(
        MAX_VRAM_GIB / TOTAL_VRAM_GIB,
        GPU_INDEX
    )

    print(f"GPU: {torch.cuda.get_device_name(GPU_INDEX)}")
    print(f"VRAM limit: {MAX_VRAM_GIB} GiB")
    print(f"CUDA device: {GPU_INDEX}")

# ============================================================
# Existing JLOW configuration
# ============================================================

BATCH_SIZE = 2
GRAD_ACCUM = 32
BLOCK_SIZE = 512

print(f"device: {DEVICE}")
print(f"batch_size: {BATCH_SIZE}")
print(f"grad_accum: {GRAD_ACCUM}")
print(f"block_size: {BLOCK_SIZE}")

# Your existing model initialization follows here:
#
# model = GPT(...)
# model.to(DEVICE)
#
# IMPORTANT:
# set_per_process_memory_fraction() must happen
# BEFORE model.to(DEVICE).
