#!/usr/bin/env python3
"""
JLOW V2 - ONE CLICK COLAB RUNNER

Flow:
1. Mount Google Drive
2. Clone/update GitHub project
3. Check required files
4. Download checkpoint-750
5. Verify files
6. Run dataset preparation
7. Backup dataset to Google Drive
8. Run training
9. Backup checkpoints to Google Drive
10. Final verification

Expected project:
    https://github.com/pelemx/jlowtrain

Expected files:
    model.py
    train.py
    train_jlow_v2.py
    prepare_jlow_v2.py
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

REPO_URL = "https://github.com/pelemx/jlowtrain"
PROJECT_DIR = Path("/content/jlowtrain")

DATA_DIR = PROJECT_DIR / "data" / "jlow_v2"
TRAIN_OUT = PROJECT_DIR / "out_jlow_v2"

DRIVE_ROOT = Path("/content/drive/MyDrive/JLOW_V2")

DRIVE_DATA = DRIVE_ROOT / "data" / "jlow_v2"
DRIVE_OUT = DRIVE_ROOT / "out_jlow_v2"
DRIVE_BASE = DRIVE_ROOT / "base_checkpoint"

BASE_CKPT = PROJECT_DIR / "out_gpt2_large" / "ckpt.pt"

# Previous checkpoint-750 Drive URL
CKPT_750_URL = (
    "https://drive.google.com/file/d/"
    "1sqYVdQH8Y6myxxePqkPC9bGuERxtr-da/view?usp=drive_link"
)

# Training limit
# Set 250 for first test run.
MAX_ITERS = 250

# Change to 5000 after validating behavior.
# MAX_ITERS = 5000


# ============================================================
# UTILS
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def run(cmd, cwd=None, check=True):
    print()
    print("$", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {cmd}"
        )

    return result.returncode


def shell(command, cwd=None, check=True):
    print()
    print("$", command)

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        shell=True,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command}"
        )

    return result.returncode


def size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def show_file(path):
    path = Path(path)

    if not path.exists():
        print(f"[MISSING] {path}")
        return False

    if path.is_file():
        print(
            f"[OK] {path} | "
            f"{size_mb(path):,.2f} MB"
        )
        return True

    print(f"[OK] {path}/")
    return True


def copy_tree(src, dst):
    src = Path(src)
    dst = Path(dst)

    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(item, target)

        print(f"BACKUP: {item} -> {target}")


# ============================================================
# DRIVE
# ============================================================

def mount_drive():
    banner("1. MOUNT GOOGLE DRIVE")

    try:
        from google.colab import drive
    except ImportError:
        raise RuntimeError(
            "This runner is intended for Google Colab."
        )

    drive.mount("/content/drive")

    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "Google Drive was not mounted correctly."
        )

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

    print("Drive:", DRIVE_ROOT)


# ============================================================
# GPU
# ============================================================

def check_gpu():
    banner("2. CHECK GPU")

    try:
        import torch
    except ImportError:
        print("PyTorch not installed yet.")
        return

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "CUDA:",
            torch.version.cuda
        )
    else:
        print(
            "WARNING: CUDA unavailable. "
            "Training will be CPU and extremely slow."
        )


# ============================================================
# GITHUB
# ============================================================

def setup_repo():
    banner("3. GET JLOW TRAINING PROJECT")

    if not PROJECT_DIR.exists():
        run([
            "git",
            "clone",
            REPO_URL,
            PROJECT_DIR,
        ])
    else:
        print("Repository already exists.")

        # Don't destroy local modifications.
        run([
            "git",
            "fetch",
            "--all",
        ], cwd=PROJECT_DIR, check=False)

    print("Project:", PROJECT_DIR)

    required = [
        "model.py",
        "train.py",
    ]

    optional = [
        "prepare_jlow_v2.py",
        "train_jlow_v2.py",
        "runner_train.py",
    ]

    for name in required:
        show_file(PROJECT_DIR / name)

    print()
    print("Optional files:")

    for name in optional:
        show_file(PROJECT_DIR / name)


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

def install_dependencies():
    banner("4. INSTALL PYTHON DEPENDENCIES")

    packages = [
        "numpy",
        "tiktoken",
        "datasets",
        "huggingface_hub",
        "gdown",
        "pyarrow",
    ]

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        *packages,
    ])


# ============================================================
# CHECKPOINT
# ============================================================

def download_checkpoint():
    banner("5. CHECK / DOWNLOAD CHECKPOINT 750")

    BASE_CKPT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BASE_CKPT.exists():
        print("Checkpoint already exists:")
        show_file(BASE_CKPT)
        return

    # First look for Drive backup.
    drive_candidates = [
        DRIVE_BASE / "ckpt.pt",
        DRIVE_BASE / "ckpt_750.pt",
        DRIVE_ROOT / "ckpt.pt",
        DRIVE_ROOT / "ckpt_750.pt",
    ]

    for candidate in drive_candidates:
        if candidate.exists():
            print(
                f"Found checkpoint in Drive: {candidate}"
            )

            shutil.copy2(
                candidate,
                BASE_CKPT,
            )

            show_file(BASE_CKPT)
            return

    print("Checkpoint not found locally.")
    print("Trying Google Drive download...")

    # Extract Drive file ID from URL.
    file_id = "1sqYVdQH8Y6myxxePqkPC9bGuERxtr-da"

    run([
        sys.executable,
        "-m",
        "gdown",
        "--id",
        file_id,
        "-O",
        BASE_CKPT,
    ], check=False)

    if not BASE_CKPT.exists():
        raise RuntimeError(
            "\n"
            "Checkpoint 750 could not be downloaded.\n\n"
            "Put ckpt.pt manually into:\n"
            f"{DRIVE_BASE}\n\n"
            "then rerun this runner."
        )

    show_file(BASE_CKPT)


# ============================================================
# CHECK FILES
# ============================================================

def verify_project():
    banner("6. VERIFY PROJECT FILES")

    files = [
        PROJECT_DIR / "model.py",
        PROJECT_DIR / "train.py",
        PROJECT_DIR / "prepare_jlow_v2.py",
        PROJECT_DIR / "train_jlow_v2.py",
        BASE_CKPT,
    ]

    missing = []

    for path in files:
        if path.exists():
            show_file(path)
        else:
            missing.append(path)

    if missing:
        print()
        print("MISSING FILES:")

        for path in missing:
            print(" -", path)

        raise RuntimeError(
            "Required JLOW files are missing from repository."
        )

    print()
    print("PROJECT CHECK: PASS")


# ============================================================
# PATCH TRAIN CONFIG
# ============================================================

def configure_training():
    banner("7. CONFIGURE JLOW V2 TRAINING")

    train_file = PROJECT_DIR / "train_jlow_v2.py"

    text = train_file.read_text(
        encoding="utf-8"
    )

    # Force local paths.
    replacements = {
        'INIT_CKPT = "/content/out_gpt2_large/ckpt.pt"':
            f'INIT_CKPT = "{BASE_CKPT}"',

        'DATA_DIR = "data/jlow_v2"':
            'DATA_DIR = "data/jlow_v2"',

        'OUT_DIR = "out_jlow_v2"':
            'OUT_DIR = "out_jlow_v2"',

        'MAX_ITERS = 5000':
            f'MAX_ITERS = {MAX_ITERS}',
    }

    changed = False

    for old, new in replacements.items():
        if old in text:
            text = text.replace(
                old,
                new,
            )
            changed = True

    train_file.write_text(
        text,
        encoding="utf-8",
    )

    if changed:
        print("Training config patched.")
    else:
        print(
            "No matching config lines changed. "
            "Check train_jlow_v2.py manually."
        )

    print(
        "MAX_ITERS:",
        MAX_ITERS,
    )


# ============================================================
# DATASET
# ============================================================

def prepare_dataset():
    banner("8. BUILD JLOW V2 DATASET")

    script = PROJECT_DIR / "prepare_jlow_v2.py"

    if not script.exists():
        raise RuntimeError(
            "prepare_jlow_v2.py not found."
        )

    # HF_TOKEN should already be present in Colab environment.
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN detected in environment.")
    else:
        print(
            "WARNING: HF_TOKEN not detected."
        )

    run([
        sys.executable,
        script,
    ], cwd=PROJECT_DIR)

    required = [
        DATA_DIR / "train.bin",
        DATA_DIR / "val.bin",
        DATA_DIR / "manifest.json",
    ]

    for path in required:
        show_file(path)

    for path in required:
        if not path.exists():
            raise RuntimeError(
                f"Dataset build incomplete: {path}"
            )

    print()
    print("DATASET BUILD: PASS")


# ============================================================
# BACKUP DATASET
# ============================================================

def backup_dataset():
    banner("9. BACKUP DATASET TO GOOGLE DRIVE")

    copy_tree(
        DATA_DIR,
        DRIVE_DATA,
    )

    print()
    print("Dataset backup complete.")

    for path in DRIVE_DATA.iterdir():
        show_file(path)


# ============================================================
# TRAIN
# ============================================================

def train():
    banner(
        f"10. START JLOW V2 TRAINING "
        f"(MAX_ITERS={MAX_ITERS})"
    )

    script = PROJECT_DIR / "train_jlow_v2.py"

    if not script.exists():
        raise RuntimeError(
            "train_jlow_v2.py not found."
        )

    print()
    print("Training locally:")
    print("DATA :", DATA_DIR)
    print("OUT  :", TRAIN_OUT)
    print("CKPT :", BASE_CKPT)
    print("ITERS:", MAX_ITERS)

    # Run normally.
    run([
        sys.executable,
        script,
    ], cwd=PROJECT_DIR)


# ============================================================
# BACKUP CHECKPOINTS
# ============================================================

def backup_training():
    banner("11. BACKUP TRAINING OUTPUT TO GOOGLE DRIVE")

    if not TRAIN_OUT.exists():
        print(
            "No training output directory found."
        )
        return

    copy_tree(
        TRAIN_OUT,
        DRIVE_OUT,
    )

    print()
    print("Training backup complete.")


# ============================================================
# FINAL REPORT
# ============================================================

def final_report():
    banner("12. JLOW V2 RUN FINISHED")

    print()
    print("LOCAL:")
    print("Project :", PROJECT_DIR)
    print("Dataset :", DATA_DIR)
    print("Output  :", TRAIN_OUT)

    print()
    print("DRIVE:")
    print("Dataset :", DRIVE_DATA)
    print("Output  :", DRIVE_OUT)

    print()
    print("Dataset files:")

    if DATA_DIR.exists():
        for path in DATA_DIR.iterdir():
            show_file(path)

    print()
    print("Training files:")

    if TRAIN_OUT.exists():
        for path in TRAIN_OUT.iterdir():
            show_file(path)

    print()
    print("DONE.")


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    try:

        banner("JLOW V2 AUTOMATED COLAB RUNNER")

        print("Repository:")
        print(REPO_URL)

        print()
        print("MAX_ITERS:")
        print(MAX_ITERS)

        # 1
        mount_drive()

        # 2
        check_gpu()

        # 3
        setup_repo()

        # 4
        install_dependencies()

        # 5
        download_checkpoint()

        # 6
        verify_project()

        # 7
        configure_training()

        # 8
        prepare_dataset()

        # 9
        backup_dataset()

        # 10
        train()

        # 11
        backup_training()

        # 12
        final_report()

        elapsed = time.time() - start

        print()
        print(
            f"TOTAL TIME: {elapsed / 3600:.2f} hours"
        )

    except KeyboardInterrupt:
        print()
        print("RUNNER INTERRUPTED.")

        # Still attempt checkpoint backup.
        try:
            backup_training()
        except Exception as e:
            print(
                "Backup after interruption failed:",
                e,
            )

        raise

    except Exception as e:
        print()
        print("=" * 70)
        print("RUNNER FAILED")
        print("=" * 70)
        print(type(e).__name__)
        print(e)

        # Attempt emergency backup.
        try:
            backup_training()
        except Exception as backup_error:
            print(
                "Emergency backup failed:",
                backup_error,
            )

        raise


if __name__ == "__main__":
    main()