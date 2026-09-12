```python
import os
import sys
import time
import shutil
import subprocess
import re
from pathlib import Path


# ============================================================
# JLOW V2 SERVER RUNNER
# RTX 6000 / PHYSICAL GPU 1
# ============================================================

# Physical GPU 1 -> logical cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================
# PATH
# ============================================================

PROJECT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT / "train_jlow_v2.py"
MODEL_FILE = PROJECT / "model.py"

DATA_DIR = PROJECT / "data" / "jlow_v2"

TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

CKPT_DIR = PROJECT / "out_gpt2_large"
CKPT_750 = CKPT_DIR / "ckpt.pt"

OUT_DIR = PROJECT / "out_jlow_v2"

LOCAL_LATEST = OUT_DIR / "ckpt_latest.pt"
LOCAL_BEST = OUT_DIR / "ckpt_best.pt"

BACKUP_DIR = PROJECT / "server_backup" / "jlow_v2"


# ============================================================
# GOOGLE DRIVE FILES
# ============================================================

CKPT_750_URL = (
    "https://drive.google.com/uc?id="
    "1sqYVdQH8Y6myxxePqK9bGuERxtr-da"
)

CKPT_BEST_URL = (
    "https://drive.google.com/uc?id="
    "1sj_zZYczQU-tk-0mtzfEp0WSjw9KNNhc"
)

CKPT_LATEST_URL = (
    "https://drive.google.com/uc?id="
    "1EWF-cEvuoLSUXHFvuq2qZubgBAKuk7ll"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

MAX_ITERS = 250

MAX_VRAM_GIB = 20
TOTAL_VRAM_GIB = 24


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def run(cmd, cwd=None, env=None):
    print()
    print(">>>", " ".join(map(str, cmd)))
    sys.stdout.flush()

    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
    )


def size_gb(path):
    return path.stat().st_size / (1024 ** 3)


def valid_file(path, minimum_bytes=100_000_000):
    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size >= minimum_bytes
    )


# ============================================================
# START
# ============================================================

header("JLOW V2 SERVER RUNNER")

print(f"PROJECT       : {PROJECT}")
print("PHYSICAL GPU  : 1")
print("VRAM LIMIT    : 20 GiB")
print(f"MAX_ITERS     : {MAX_ITERS}")
print()
print("Other GPU processes will NOT be touched.")
print("GPU 0 will NOT be touched.")
print("OmniVoice will NOT be touched.")


# ============================================================
# [1/10] GPU CHECK
# ============================================================

header("[1/10] GPU CHECK")

run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free",
    "--format=csv"
])


# ============================================================
# [2/10] PYTORCH CHECK
# ============================================================

header("[2/10] PYTORCH / CUDA CHECK")

gpu_test = r'''
import torch

print("torch        :", torch.__version__)
print("cuda         :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA tidak tersedia.")

print("cuda version :", torch.version.cuda)
print("device count :", torch.cuda.device_count())

idx = torch.cuda.current_device()

print("logical cuda :", idx)
print("GPU          :", torch.cuda.get_device_name(idx))

props = torch.cuda.get_device_properties(idx)

print(
    "VRAM total   :",
    round(props.total_memory / (1024 ** 3), 2),
    "GiB"
)

torch.cuda.set_per_process_memory_fraction(
    20 / 24,
    idx,
)

print("VRAM limit   : 20 GiB")
'''

run(
    [
        sys.executable,
        "-c",
        gpu_test,
    ],
    env=os.environ.copy(),
)


# ============================================================
# [3/10] CHECK PROJECT FILES
# ============================================================

header("[3/10] PROJECT CHECK")

for path in [
    MODEL_FILE,
    TRAIN_FILE,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    print(
        f"OK: {path.relative_to(PROJECT)}"
    )


# ============================================================
# [4/10] INSTALL / CHECK GDOWN
# ============================================================

header("[4/10] GOOGLE DRIVE DOWNLOADER")

try:
    import gdown

    print("gdown already installed.")

except ImportError:

    print("Installing gdown...")

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "gdown",
    ])


# ============================================================
# [5/10] DOWNLOAD CHECKPOINT 750
# ============================================================

header("[5/10] CHECKPOINT 750")

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if valid_file(CKPT_750):

    print("Checkpoint 750 already exists:")
    print(CKPT_750)
    print(f"Size: {size_gb(CKPT_750):.3f} GB")

else:

    print("Checkpoint 750 not found.")
    print("Downloading from Google Drive...")

    run([
        sys.executable,
        "-m",
        "gdown",
        CKPT_750_URL,
        "-O",
        str(CKPT_750),
    ])

    if not valid_file(CKPT_750):
        raise RuntimeError(
            "Checkpoint 750 gagal didownload "
            "atau file terlalu kecil."
        )

    print(
        f"Downloaded: {size_gb(CKPT_750):.3f} GB"
    )


# ============================================================
# [6/10] DATASET CHECK
# ============================================================

header("[6/10] DATASET CHECK")

if not TRAIN_BIN.exists():
    raise FileNotFoundError(
        f"train.bin tidak ditemukan:\n{TRAIN_BIN}\n\n"
        "Dataset JLOW V2 harus tersedia di server."
    )

if not VAL_BIN.exists():
    raise FileNotFoundError(
        f"val.bin tidak ditemukan:\n{VAL_BIN}\n\n"
        "Dataset JLOW V2 harus tersedia di server."
    )

print(
    f"train.bin : {size_gb(TRAIN_BIN):.3f} GB"
)

print(
    f"val.bin   : {size_gb(VAL_BIN):.3f} GB"
)

print(
    f"train tokens ~ {TRAIN_BIN.stat().st_size // 2:,}"
)

print(
    f"val tokens   ~ {VAL_BIN.stat().st_size // 2:,}"
)


# ============================================================
# [7/10] RESTORE V2 CHECKPOINT FROM GOOGLE DRIVE
# ============================================================

header("[7/10] V2 CHECKPOINT RESTORE")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# IMPORTANT:
#
# If local ckpt_latest exists:
#     keep it
#
# Otherwise:
#     download V2 latest from Drive
#
# If Drive latest does not work:
#     try best
#
# If neither exists:
#     start from checkpoint 750
# ------------------------------------------------------------

if valid_file(LOCAL_LATEST):

    print("Local V2 latest checkpoint found.")
    print(f"Using: {LOCAL_LATEST}")
    print(f"Size : {size_gb(LOCAL_LATEST):.3f} GB")

else:

    print("Local V2 latest checkpoint not found.")
    print("Checking Google Drive V2 latest...")

    try:

        run([
            sys.executable,
            "-m",
            "gdown",
            CKPT_LATEST_URL,
            "-O",
            str(LOCAL_LATEST),
        ])

    except subprocess.CalledProcessError:

        print(
            "WARNING: Drive latest download failed."
        )

    if valid_file(LOCAL_LATEST):

        print(
            f"V2 latest downloaded: "
            f"{size_gb(LOCAL_LATEST):.3f} GB"
        )

    else:

        print(
            "V2 latest unavailable."
        )

        # Remove broken partial file.
        if LOCAL_LATEST.exists():
            LOCAL_LATEST.unlink()


# ------------------------------------------------------------
# Optional best checkpoint
# ------------------------------------------------------------

if valid_file(LOCAL_BEST):

    print(
        f"Local best checkpoint exists: "
        f"{LOCAL_BEST}"
    )

else:

    print("Local best checkpoint not found.")

    try:

        run([
            sys.executable,
            "-m",
            "gdown",
            CKPT_BEST_URL,
            "-O",
            str(LOCAL_BEST),
        ])

    except subprocess.CalledProcessError:

        print(
            "WARNING: Drive best download failed."
        )

    if not valid_file(LOCAL_BEST):

        if LOCAL_BEST.exists():
            LOCAL_BEST.unlink()

        print(
            "Best checkpoint unavailable."
        )


# ============================================================
# DETERMINE TRAINING MODE
# ============================================================

if valid_file(LOCAL_LATEST):

    TRAIN_MODE = "RESUME_V2"

else:

    TRAIN_MODE = "START_FROM_750"


print()
print("TRAINING MODE:")
print(f"  {TRAIN_MODE}")


# ============================================================
# [8/10] CONFIGURE TRAIN SCRIPT
# ============================================================

header("[8/10] CONFIGURE TRAINING")

text = TRAIN_FILE.read_text()


# ------------------------------------------------------------
# INIT_CKPT
# ------------------------------------------------------------

text, count = re.subn(
    r'^INIT_CKPT\s*=\s*.*$',
    f'INIT_CKPT = "{CKPT_750}"',
    text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "INIT_CKPT tidak ditemukan."
    )


# ------------------------------------------------------------
# DATA_DIR
# ------------------------------------------------------------

text, count = re.subn(
    r'^DATA_DIR\s*=\s*.*$',
    f'DATA_DIR = "{DATA_DIR}"',
    text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "DATA_DIR tidak ditemukan."
    )


# ------------------------------------------------------------
# OUT_DIR
# ------------------------------------------------------------

text, count = re.subn(
    r'^OUT_DIR\s*=\s*.*$',
    f'OUT_DIR = "{OUT_DIR}"',
    text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "OUT_DIR tidak ditemukan."
    )


# ------------------------------------------------------------
# MAX_ITERS
# ------------------------------------------------------------

text, count = re.subn(
    r'^MAX_ITERS\s*=\s*.*$',
    f'MAX_ITERS = {MAX_ITERS}',
    text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "MAX_ITERS tidak ditemukan."
    )


# ------------------------------------------------------------
# Ensure RESUME_V2 = True
# ------------------------------------------------------------

text, count = re.subn(
    r'^RESUME_V2\s*=\s*.*$',
    'RESUME_V2 = True',
    text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "RESUME_V2 tidak ditemukan."
    )


# ------------------------------------------------------------
# Ensure exact MAX_ITERS ending
# ------------------------------------------------------------

text = text.replace(
    "if iter_num > MAX_ITERS:",
    "if iter_num >= MAX_ITERS:"
)


TRAIN_FILE.write_text(text)


print("Training configuration:")
print(f"  INIT_CKPT = {CKPT_750}")
print(f"  DATA_DIR  = {DATA_DIR}")
print(f"  OUT_DIR   = {OUT_DIR}")
print(f"  MAX_ITERS = {MAX_ITERS}")
print("  RESUME_V2 = True")


# ============================================================
# [9/10] START TRAINING
# ============================================================

header("[9/10] START TRAINING")

env = os.environ.copy()

# Physical GPU 1
env["CUDA_VISIBLE_DEVICES"] = "1"

print()
print("GPU mapping:")
print("  physical GPU : 1")
print("  logical CUDA  : 0")
print("  VRAM limit    : 20 GiB")
print()
print("Other processes are untouched.")
print()

sys.stdout.flush()

start_time = time.time()

process = subprocess.run(
    [
        sys.executable,
        "-u",
        str(TRAIN_FILE),
    ],
    cwd=PROJECT,
    env=env,
)

if process.returncode != 0:

    raise RuntimeError(
        f"Training gagal. Exit code: "
        f"{process.returncode}"
    )

elapsed = time.time() - start_time

print()
print(
    f"Training runtime: "
    f"{elapsed / 3600:.2f} hours"
)


# ============================================================
# [10/10] BACKUP
# ============================================================

header("[10/10] BACKUP OUTPUT")

if not OUT_DIR.exists():
    raise RuntimeError(
        "out_jlow_v2 tidak ditemukan setelah training."
    )


BACKUP_DIR.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if BACKUP_DIR.exists():
    shutil.rmtree(BACKUP_DIR)

shutil.copytree(
    OUT_DIR,
    BACKUP_DIR,
)

print(
    f"Backup created:\n{BACKUP_DIR}"
)

print()
print("Checkpoint files:")

for path in sorted(OUT_DIR.glob("*.pt")):

    print(
        f"  {path.name:<20}"
        f" {size_gb(path):.3f} GB"
    )


# ============================================================
# FINAL GPU STATUS
# ============================================================

header("FINAL GPU STATUS")

try:

    run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv",
    ])

except Exception:

    print(
        "WARNING: nvidia-smi final check gagal."
    )


# ============================================================
# DONE
# ============================================================

header("JLOW V2 SERVER RUNNER FINISHED")

print("SUCCESS")
print()
print(f"Project : {PROJECT}")
print(f"Output  : {OUT_DIR}")
print(f"Backup  : {BACKUP_DIR}")
print()
print("Physical GPU 1 used.")
print("PyTorch VRAM limit: 20 GiB.")
print("GPU 0 untouched.")
print("Other processes untouched.")
```
