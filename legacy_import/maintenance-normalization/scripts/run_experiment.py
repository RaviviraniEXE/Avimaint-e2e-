"""One-shot experiment runner: prepare -> dictionary -> split -> normalize -> evaluate."""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys

PY = sys.executable


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D"])
    ap.add_argument("--byt5-dir", default=None)
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()
    sh([PY, "scripts/01_prepare_data.py"])
    sh([PY, "scripts/02_build_dictionary.py"])
    sh([PY, "scripts/split_gold.py"])
    cmd = [PY, "scripts/03_run_normalization.py", "--systems", *args.systems]
    if args.byt5_dir:
        cmd += ["--byt5-dir", args.byt5_dir]
    sh(cmd)
    sh([PY, "scripts/04_evaluate.py", "--run-id", args.run_id])
    print("\nDone. See outputs/reports/experiment_report.md")


if __name__ == "__main__":
    main()

