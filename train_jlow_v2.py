import os
import time
import math
import torch
import numpy as np
from contextlib import nullcontext

from model import GPTConfig, GPT


# ============================================================
# JLOW V2 TRAINING
# Base checkpoint: /content/out_gpt2_large/ckpt.pt
# Dataset:        data/jlow_v2
#
# First run:
#   checkpoint 750 -> LOAD WEIGHTS ONLY
#   optimizer     -> RESET
#   iter          -> 0
#
# Later runs:
#   out_jlow_v2/ckpt_latest.pt -> resume V2
# ============================================================

INIT_CKPT = "/content/out_gpt2_large/ckpt.pt"
DATA_DIR = "data/jlow_v2"
OUT_DIR = "out_jlow_v2"

BLOCK_SIZE = 512

BATCH_SIZE = 2
GRAD_ACCUM = 32

MAX_ITERS = 250

LEARNING_RATE = 1e-5
MIN_LR = 1e-6
WARMUP_ITERS = 200

WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_ITERS = 30
LOG_INTERVAL = 10

SEED = 1337

RESUME_V2 = True
USE_COMPILE = False


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.manual_seed(SEED)

device_type = "cuda" if "cuda" in DEVICE else "cpu"

if device_type == "cuda":
    if torch.cuda.is_bf16_supported():
        DTYPE = "bfloat16"
    else:
        DTYPE = "float16"
else:
    DTYPE = "float32"

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]

ctx = (
    torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)

print("=" * 70)
print("JLOW V2 TRAINING")
print("=" * 70)
print(f"device        : {DEVICE}")
print(f"dtype         : {DTYPE}")
print(f"dataset       : {DATA_DIR}")
print(f"base checkpoint: {INIT_CKPT}")
print(f"output        : {OUT_DIR}")
print(f"block_size    : {BLOCK_SIZE}")
print(f"batch_size    : {BATCH_SIZE}")
print(f"grad_accum    : {GRAD_ACCUM}")
print(f"effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"max_iters     : {MAX_ITERS}")
print(f"learning_rate : {LEARNING_RATE}")
print("=" * 70)


# ============================================================
# DATA CHECK
# ============================================================

train_path = os.path.join(DATA_DIR, "train.bin")
val_path = os.path.join(DATA_DIR, "val.bin")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"train.bin tidak ditemukan: {train_path}")

if not os.path.exists(val_path):
    raise FileNotFoundError(f"val.bin tidak ditemukan: {val_path}")


def get_batch(split):
    path = train_path if split == "train" else val_path

    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    max_start = len(data) - BLOCK_SIZE - 1

    if max_start <= 0:
        raise RuntimeError(
            f"Dataset {split} terlalu pendek untuk block_size={BLOCK_SIZE}"
        )

    ix = torch.randint(max_start, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(
            data[i:i + BLOCK_SIZE].astype(np.int64)
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)
        )
        for i in ix
    ])

    if device_type == "cuda":
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

    return x, y


# ============================================================
# CHECKPOINT LOAD
# ============================================================

v2_latest = os.path.join(OUT_DIR, "ckpt_latest.pt")

iter_num = 0
best_val_loss = float("inf")
resume_optimizer_state = None


if RESUME_V2 and os.path.exists(v2_latest):

    print()
    print("Existing V2 checkpoint found.")
    print(f"Resuming V2 from: {v2_latest}")

    checkpoint = torch.load(
        v2_latest,
        map_location="cpu",
        weights_only=False,
    )

    model_args = dict(checkpoint["model_args"])

    # V2 always uses this sequence length.
    model_args["block_size"] = BLOCK_SIZE

    iter_num = int(checkpoint["iter_num"])
    best_val_loss = float(checkpoint["best_val_loss"])

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=True)

    resume_optimizer_state = checkpoint.get("optimizer")

    print(f"resume iter : {iter_num}")
    print(f"best val    : {best_val_loss}")

else:

    print()
    print("Starting JLOW V2 from checkpoint 750.")
    print("Loading WEIGHTS ONLY.")
    print("Optimizer will be RESET.")
    print("Iteration will start at 0.")
    print()

    if not os.path.exists(INIT_CKPT):
        raise FileNotFoundError(
            f"Checkpoint 750 tidak ditemukan:\n{INIT_CKPT}"
        )

    checkpoint = torch.load(
        INIT_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_model_args = checkpoint["model_args"]

    print("Checkpoint architecture:")
    for k in [
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
    ]:
        print(f"  {k}: {checkpoint_model_args.get(k)}")

    expected = {
        "vocab_size": 50257,
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "bias": True,
    }

    for key, expected_value in expected.items():
        actual = checkpoint_model_args.get(key)

        if actual != expected_value:
            raise RuntimeError(
                f"CHECKPOINT ARCHITECTURE MISMATCH: "
                f"{key}: expected={expected_value}, got={actual}"
            )

    model_args = dict(checkpoint_model_args)
    model_args["block_size"] = BLOCK_SIZE

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=True)

    # IMPORTANT:
    # checkpoint["optimizer"] is intentionally NOT loaded.
    resume_optimizer_state = None
    iter_num = 0
    best_val_loss = float("inf")

    print()
    print("Checkpoint 750 weights loaded successfully.")
    print("Optimizer RESET.")
    print("Iteration RESET to 0.")


# Free checkpoint object before training.
checkpoint = None


# ============================================================
# MODEL
# ============================================================

model.to(DEVICE)

print()
print("Model:")
print(model.config)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = model.configure_optimizers(
    WEIGHT_DECAY,
    LEARNING_RATE,
    (BETA1, BETA2),
    device_type,
)

if resume_optimizer_state is not None:
    print("Loading V2 optimizer state...")
    optimizer.load_state_dict(resume_optimizer_state)


# ============================================================
# AMP
# ============================================================

scaler = torch.cuda.amp.GradScaler(
    enabled=(
        device_type == "cuda"
        and DTYPE == "float16"
    )
)


# ============================================================
# OPTIONAL COMPILE
# ============================================================

if USE_COMPILE:
    print("Compiling model...")
    model = torch.compile(model)


def get_raw_model():
    raw = model

    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    return raw


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(tag, current_iter, val_loss):

    raw_model = get_raw_model()

    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": current_iter,
        "best_val_loss": best_val_loss,
        "config": {
            "dataset": DATA_DIR,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "max_iters": MAX_ITERS,
        },
    }

    path = os.path.join(
        OUT_DIR,
        f"ckpt_{tag}.pt",
    )

    torch.save(ckpt, path)

    print(f"saved checkpoint: {path}")


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    out = {}

    for split in ["train", "val"]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            X, Y = get_batch(split)

            with ctx:
                logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(it):

    if it < WARMUP_ITERS:
        return LEARNING_RATE * (
            (it + 1) / (WARMUP_ITERS + 1)
        )

    if it >= MAX_ITERS:
        return MIN_LR

    decay_ratio = (
        (it - WARMUP_ITERS)
        / (MAX_ITERS - WARMUP_ITERS)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return MIN_LR + coeff * (
        LEARNING_RATE - MIN_LR
    )


# ============================================================
# TRAINING
# ============================================================

X, Y = get_batch("train")

t0 = time.time()

print()
print("=" * 70)
print("TRAINING START")
print("=" * 70)
print(f"starting iteration: {iter_num}")
print()

while True:

    lr = get_lr(iter_num)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # --------------------------------------------------------
    # EVAL + CHECKPOINT
    # --------------------------------------------------------

    if iter_num % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"step {iter_num}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, "
            f"lr {lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                "best",
                iter_num,
                val_loss,
            )

        save_checkpoint(
            "latest",
            iter_num,
            val_loss,
        )

    # --------------------------------------------------------
    # ZERO GRAD
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # GRADIENT ACCUMULATION
    # --------------------------------------------------------

    for micro_step in range(GRAD_ACCUM):

        with ctx:

            logits, loss = model(X, Y)

            loss = loss / GRAD_ACCUM

        X, Y = get_batch("train")

        scaler.scale(loss).backward()

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    if GRAD_CLIP != 0:

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    scaler.step(optimizer)
    scaler.update()

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % LOG_INTERVAL == 0:

        lossf = loss.item() * GRAD_ACCUM

        print(
            f"iter {iter_num}: "
            f"loss {lossf:.4f}, "
            f"lr {lr:.2e}, "
            f"time {dt:.2f}s"
        )

    iter_num += 1

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    if iter_num > MAX_ITERS:

        print()
        print("=" * 70)
        print("TRAINING FINISHED")
        print("=" * 70)

        save_checkpoint(
            "final",
            iter_num,
            best_val_loss,
        )

        break
