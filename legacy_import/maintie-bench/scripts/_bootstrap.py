import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- quiet, readable terminal output ---------------------------------------
# Silence the library deprecation / HuggingFace-hub warnings that otherwise
# bury the actual training progress. Must run BEFORE transformers is imported.
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_T0 = time.time()


def _elapsed():
    s = int(time.time() - _T0)
    return f"{s // 60:d}m{s % 60:02d}s"


def banner(msg, char="="):
    """Clear section header so it's obvious what stage is running, and when."""
    line = char * 66
    print(f"\n{line}\n  {msg:<50}[t+{_elapsed()}]\n{line}", flush=True)


def step(msg):
    """One-line progress note, timestamped."""
    print(f"  -> {msg}   [t+{_elapsed()}]", flush=True)

