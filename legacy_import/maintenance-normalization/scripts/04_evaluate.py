"""Step 4 — evaluate all systems; save reports + append to results_log.csv."""
import _bootstrap  # noqa: F401
import argparse

from src.evaluation.evaluate import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()
    res = run(run_id=args.run_id)
    if res["intrinsic"].empty:
        print("No normalized outputs. Run scripts/03_run_normalization.py first.")
        return
    print("\n=== Intrinsic (all records) ===")
    print(res["intrinsic"].to_string(index=False))
    if not res["extrinsic"].empty:
        print("\n=== Extrinsic (vs Amin gold) ===")
        print(res["extrinsic"].to_string(index=False))
    print("\nReports + results_log.csv saved in outputs/reports/")


if __name__ == "__main__":
    main()

