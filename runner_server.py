# JLOW V2 SERVER RUNNER - background + log
import os, sys, time, shutil, subprocess, re
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PROJECT = Path(__file__).resolve().parent
LOG_DIR = PROJECT / 'logs'
LOG_FILE = LOG_DIR / 'jlow_v2_runner.log'
PID_FILE = LOG_DIR / 'jlow_v2_runner.pid'
TRAIN_FILE = PROJECT / 'train_jlow_v2.py'
MODEL_FILE = PROJECT / 'model.py'
DATA_DIR = PROJECT / 'data' / 'jlow_v2'
TRAIN_BIN = DATA_DIR / 'train.bin'
VAL_BIN = DATA_DIR / 'val.bin'
CKPT_DIR = PROJECT / 'out_gpt2_large'
CKPT_750 = CKPT_DIR / 'ckpt.pt'
OUT_DIR = PROJECT / 'out_jlow_v2'
LOCAL_LATEST = OUT_DIR / 'ckpt_latest.pt'
LOCAL_BEST = OUT_DIR / 'ckpt_best.pt'
BACKUP_DIR = PROJECT / 'server_backup' / 'jlow_v2'

CKPT_750_URL = 'https://drive.google.com/uc?id=1sqYVdQH8Y6myxxePqkPC9bGuERxtr-da'
CKPT_BEST_URL = 'https://drive.google.com/uc?id=1sj_zZYczQU-tk-0mtzfEp0WSjw9KNNhc'
CKPT_LATEST_URL = 'https://drive.google.com/uc?id=1EWF-cEvuoLSUXHFvuq2qZubgBAKuk7ll'
MAX_ITERS = 250


def daemonize():
    if os.environ.get('JLOW_RUNNER_BACKGROUND') == '1':
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f'JLOW runner sudah berjalan. PID: {pid}')
            print(f'Log: {LOG_FILE}')
            print(f'Follow: tail -f {LOG_FILE}')
            raise SystemExit(0)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            PID_FILE.unlink(missing_ok=True)
    log = open(LOG_FILE, 'a', buffering=1, encoding='utf-8')
    env = os.environ.copy()
    env['JLOW_RUNNER_BACKGROUND'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = '1'
    child = subprocess.Popen(
        [sys.executable, '-u', str(Path(__file__).resolve())],
        cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
    PID_FILE.write_text(str(child.pid))
    print('=' * 72)
    print('JLOW V2 RUNNER STARTED IN BACKGROUND')
    print('=' * 72)
    print(f'PID : {child.pid}')
    print(f'LOG : {LOG_FILE}')
    print(f'Live: tail -f {LOG_FILE}')
    print(f'Status: ps -fp {child.pid}')
    log.close()
    raise SystemExit(0)


daemonize()


def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)
    sys.stdout.flush()


def run(cmd, cwd=None, env=None):
    print('\n>>>', ' '.join(map(str, cmd)))
    sys.stdout.flush()
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def size_gb(path):
    return path.stat().st_size / (1024 ** 3)


def valid_file(path, minimum_bytes=100_000_000):
    return path.exists() and path.is_file() and path.stat().st_size >= minimum_bytes


def cleanup_pid():
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


try:
    header('JLOW V2 SERVER RUNNER')
    print(f'PROJECT       : {PROJECT}')
    print('PHYSICAL GPU  : 1')
    print('VRAM LIMIT    : 20 GiB')
    print(f'MAX_ITERS     : {MAX_ITERS}')
    print(f'LOG FILE      : {LOG_FILE}')
    print(f'PID           : {os.getpid()}')
    print('Other GPU processes will NOT be touched.')
    print('GPU 0 will NOT be touched.')
    print('OmniVoice will NOT be touched.')

    header('[1/10] GPU CHECK')
    run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv'])

    header('[2/10] PYTORCH / CUDA CHECK')
    gpu_test = r'''
import torch
print('torch        :', torch.__version__)
print('cuda         :', torch.cuda.is_available())
if not torch.cuda.is_available(): raise RuntimeError('CUDA tidak tersedia.')
print('cuda version :', torch.version.cuda)
print('device count :', torch.cuda.device_count())
idx = torch.cuda.current_device()
print('logical cuda :', idx)
print('GPU          :', torch.cuda.get_device_name(idx))
props = torch.cuda.get_device_properties(idx)
print('VRAM total   :', round(props.total_memory / (1024 ** 3), 2), 'GiB')
torch.cuda.set_per_process_memory_fraction(20 / 24, idx)
print('VRAM limit   : 20 GiB')
'''
    run([sys.executable, '-c', gpu_test], env=os.environ.copy())

    header('[3/10] PROJECT CHECK')
    for path in [MODEL_FILE, TRAIN_FILE]:
        if not path.exists(): raise FileNotFoundError(f'File tidak ditemukan: {path}')
        print(f'OK: {path.relative_to(PROJECT)}')

    header('[4/10] GOOGLE DRIVE DOWNLOADER')
    try:
        import gdown
        print('gdown already installed.')
    except ImportError:
        print('Installing gdown...')
        run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])

    header('[5/10] CHECKPOINT 750')
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if valid_file(CKPT_750):
        print('Checkpoint 750 already exists:', CKPT_750)
        print(f'Size: {size_gb(CKPT_750):.3f} GB')
    else:
        print('Checkpoint 750 not found. Downloading from Google Drive...')
        run([sys.executable, '-m', 'gdown', CKPT_750_URL, '-O', str(CKPT_750)])
        if not valid_file(CKPT_750): raise RuntimeError('Checkpoint 750 gagal didownload atau file terlalu kecil.')
        print(f'Downloaded: {size_gb(CKPT_750):.3f} GB')

    header('[6/10] DATASET CHECK')
    if not TRAIN_BIN.exists(): raise FileNotFoundError(f'train.bin tidak ditemukan: {TRAIN_BIN}\nDataset JLOW V2 harus tersedia di server.')
    if not VAL_BIN.exists(): raise FileNotFoundError(f'val.bin tidak ditemukan: {VAL_BIN}\nDataset JLOW V2 harus tersedia di server.')
    print(f'train.bin : {size_gb(TRAIN_BIN):.3f} GB')
    print(f'val.bin   : {size_gb(VAL_BIN):.3f} GB')
    print(f'train tokens ~ {TRAIN_BIN.stat().st_size // 2:,}')
    print(f'val tokens   ~ {VAL_BIN.stat().st_size // 2:,}')

    header('[7/10] V2 CHECKPOINT RESTORE')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if valid_file(LOCAL_LATEST):
        print('Local V2 latest checkpoint found.')
        print(f'Using: {LOCAL_LATEST}')
        print(f'Size : {size_gb(LOCAL_LATEST):.3f} GB')
    else:
        print('Local V2 latest checkpoint not found. Checking Google Drive V2 latest...')
        try:
            run([sys.executable, '-m', 'gdown', CKPT_LATEST_URL, '-O', str(LOCAL_LATEST)])
        except subprocess.CalledProcessError:
            print('WARNING: Drive latest download failed.')
        if valid_file(LOCAL_LATEST):
            print(f'V2 latest downloaded: {size_gb(LOCAL_LATEST):.3f} GB')
        else:
            print('V2 latest unavailable.')
            if LOCAL_LATEST.exists(): LOCAL_LATEST.unlink()

    if valid_file(LOCAL_BEST):
        print(f'Local best checkpoint exists: {LOCAL_BEST}')
    else:
        print('Local best checkpoint not found.')
        try:
            run([sys.executable, '-m', 'gdown', CKPT_BEST_URL, '-O', str(LOCAL_BEST)])
        except subprocess.CalledProcessError:
            print('WARNING: Drive best download failed.')
        if not valid_file(LOCAL_BEST):
            if LOCAL_BEST.exists(): LOCAL_BEST.unlink()
            print('Best checkpoint unavailable.')

    TRAIN_MODE = 'RESUME_V2' if valid_file(LOCAL_LATEST) else 'START_FROM_750'
    print('\nTRAINING MODE:')
    print(f'  {TRAIN_MODE}')

    header('[8/10] CONFIGURE TRAINING')
    text = TRAIN_FILE.read_text()
    patterns = [
        (r'^INIT_CKPT\s*=\s*.*$', f'INIT_CKPT = "{CKPT_750}"', 'INIT_CKPT'),
        (r'^DATA_DIR\s*=\s*.*$', f'DATA_DIR = "{DATA_DIR}"', 'DATA_DIR'),
        (r'^OUT_DIR\s*=\s*.*$', f'OUT_DIR = "{OUT_DIR}"', 'OUT_DIR'),
        (r'^MAX_ITERS\s*=\s*.*$', f'MAX_ITERS = {MAX_ITERS}', 'MAX_ITERS'),
        (r'^RESUME_V2\s*=\s*.*$', 'RESUME_V2 = True', 'RESUME_V2'),
    ]
    for pat, repl, name in patterns:
        text, count = re.subn(pat, repl, text, count=1, flags=re.MULTILINE)
        if count != 1: raise RuntimeError(f'{name} tidak ditemukan.')
    text = text.replace('if iter_num > MAX_ITERS:', 'if iter_num >= MAX_ITERS:')
    TRAIN_FILE.write_text(text)
    print(f'INIT_CKPT = {CKPT_750}')
    print(f'DATA_DIR  = {DATA_DIR}')
    print(f'OUT_DIR   = {OUT_DIR}')
    print(f'MAX_ITERS = {MAX_ITERS}')
    print('RESUME_V2 = True')

    header('[9/10] START TRAINING')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    print('GPU mapping: physical GPU 1 -> logical CUDA 0')
    print('VRAM limit: 20 GiB')
    print('Other processes are untouched.')
    print('TRAINING START')
    sys.stdout.flush()
    start_time = time.time()
    process = subprocess.run([sys.executable, '-u', str(TRAIN_FILE)], cwd=PROJECT, env=env)
    if process.returncode != 0: raise RuntimeError(f'Training gagal. Exit code: {process.returncode}')
    print(f'Training runtime: {(time.time() - start_time) / 3600:.2f} hours')

    header('[10/10] BACKUP OUTPUT')
    if not OUT_DIR.exists(): raise RuntimeError('out_jlow_v2 tidak ditemukan setelah training.')
    BACKUP_DIR.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP_DIR.exists(): shutil.rmtree(BACKUP_DIR)
    shutil.copytree(OUT_DIR, BACKUP_DIR)
    print(f'Backup created: {BACKUP_DIR}')
    print('Checkpoint files:')
    for path in sorted(OUT_DIR.glob('*.pt')): print(f'  {path.name:<20} {size_gb(path):.3f} GB')

    header('FINAL GPU STATUS')
    try: run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv'])
    except Exception: print('WARNING: nvidia-smi final check gagal.')

    header('JLOW V2 SERVER RUNNER FINISHED')
    print('SUCCESS')
    print(f'Project : {PROJECT}')
    print(f'Output  : {OUT_DIR}')
    print(f'Backup  : {BACKUP_DIR}')
    print(f'Log     : {LOG_FILE}')
    print('Physical GPU 1 used.')
    print('PyTorch VRAM limit: 20 GiB.')
    print('GPU 0 untouched.')
    print('Other processes untouched.')
finally:
    cleanup_pid()
