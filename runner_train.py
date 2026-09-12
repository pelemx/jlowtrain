
import os
import sys
import subprocess
from pathlib import Path
import shutil

print("=" * 70, flush=True)
print("JLOW V2 AUTOMATED RUNNER", flush=True)
print("=" * 70, flush=True)

REPO = "https://github.com/pelemx/jlowtrain"
PROJECT = Path("/content/jlowtrain")
DRIVE = Path("/content/drive/MyDrive/JLOW_V2")
DATA = PROJECT / "data/jlow_v2"
OUT = PROJECT / "out_jlow_v2"
CKPT = PROJECT / "out_gpt2_large/ckpt.pt"

MAX_ITERS = 250


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


print("[1/9] CHECK DRIVE", flush=True)

if not Path("/content/drive/MyDrive").exists():
    raise RuntimeError(
        "Google Drive belum mounted. Jalankan drive.mount() di cell Colab dulu."
    )

DRIVE.mkdir(parents=True, exist_ok=True)


print("[2/9] CHECK GPU", flush=True)

try:
    run(["nvidia-smi"])
except Exception:
    print("WARNING: nvidia-smi gagal.", flush=True)


print("[4/9] DOWNLOAD CHECKPOINT 750", flush=True)

CKPT_URL = "https://drive.google.com/uc?id=1sqYVdQH8Y6myxxePqkPC9bGuERxtr-da"
CKPT.parent.mkdir(parents=True, exist_ok=True)

run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "gdown"
])

run([
    sys.executable,
    "-m",
    "gdown",
    CKPT_URL,
    "-O",
    str(CKPT)
])

if not CKPT.exists():
    raise FileNotFoundError("Checkpoint gagal didownload.")

size_gb = CKPT.stat().st_size / (1024 ** 3)

print(
    f"Checkpoint downloaded: {CKPT}",
    flush=True
)
print(
    f"Checkpoint size: {size_gb:.2f} GB",
    flush=True
)

if CKPT.stat().st_size < 100_000_000:
    raise RuntimeError(
        "File checkpoint terlalu kecil. Kemungkinan Google Drive "
        "mengembalikan halaman konfirmasi/error, bukan ckpt.pt."
    )

# Backup langsung ke Drive
drive_ckpt = DRIVE / "base_checkpoint/ckpt.pt"
drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(CKPT, drive_ckpt)

print(f"Checkpoint backup: {drive_ckpt}", flush=True)

print("[5/9] PREPARE JLOW V2 DATASET", flush=True)

run(
    [sys.executable, "-u", "prepare_jlow_v2.py"],
    cwd=PROJECT
)


print("[6/9] BACKUP DATASET -> DRIVE", flush=True)

drive_data = DRIVE / "data/jlow_v2"

if drive_data.exists():
    shutil.rmtree(drive_data)

shutil.copytree(DATA, drive_data)

print(f"Dataset backed up to: {drive_data}", flush=True)


print("[7/9] CONFIG TRAINING", flush=True)

train_file = PROJECT / "train_jlow_v2.py"

text = train_file.read_text()

text = text.replace(
    'INIT_CKPT = "/content/out_gpt2_large/ckpt.pt"',
    f'INIT_CKPT = "{CKPT}"'
)

text = text.replace(
    "MAX_ITERS = 5000",
    f"MAX_ITERS = {MAX_ITERS}"
)

train_file.write_text(text)

print(f"MAX_ITERS = {MAX_ITERS}", flush=True)
print(f"INIT_CKPT = {CKPT}", flush=True)


print("[8/9] START TRAINING", flush=True)

run(
    [sys.executable, "-u", "train_jlow_v2.py"],
    cwd=PROJECT
)


print("[9/9] BACKUP TRAINING OUTPUT -> DRIVE", flush=True)

drive_out = DRIVE / "out_jlow_v2"

if drive_out.exists():
    shutil.rmtree(drive_out)

if OUT.exists():
    shutil.copytree(OUT, drive_out)

print(f"Training output backed up to: {drive_out}", flush=True)

print("\n" + "=" * 70, flush=True)
print("JLOW V2 RUNNER FINISHED", flush=True)
print("=" * 70, flush=True)
