"""Step 1 — make the RANDOM pilot from raw data and export it for Label Studio.

raw logbook -> dedup -> RANDOM sample (unbiased) -> weak pre-annotation ->
Label Studio tasks (+ labeling-config XML) + reports.

  python scripts/01_make_pilot.py            # uses annotation.pilot_size from config
  python scripts/01_make_pilot.py --n 300
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

from src.data.batching import write_batches
from src.data.corpus import annotated_idents, load
from src.data.labelstudio import labeling_config, to_tasks
from src.data.preannotate import coverage, preannotate
from src.data.sampling import describe, random_sample
from src.schema import bio_tags, load_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    schema = load_schema()
    ann = schema["annotation"]
    n = args.n or ann["pilot_size"]

    df, pool, norm_map, dstats = load()
    already = annotated_idents()
    sample = random_sample(pool, n, ann["seed"], exclude=already)

    records = []
    for _, r in sample.iterrows():
        pa = preannotate(r["text"])
        pa.update({"ident": r["IDENT"], "stratum": "pilot_random",
                   "exact_group_id": r["exact_group_id"],
                   "problem_raw": r["PROBLEM"], "action_raw": r["ACTION"]})
        records.append(pa)

    os.makedirs("outputs/pilot", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    # Label Studio import tasks (+ human-review txt batches)
    tasks = to_tasks(records)
    json.dump(tasks, open("outputs/pilot/pilot_tasks.json", "w"), ensure_ascii=False, indent=1)
    write_batches(records, "outputs/pilot", "pilot", ann["batch_size"])
    open("outputs/labeling_config.xml", "w").write(labeling_config())

    audit = {"records": len(df), **dstats,
             "pilot_size": len(sample), "pilot_composition_random": describe(sample)}
    report = {"dataset_audit": audit,
              "preannotation_coverage": coverage(records),
              "bio_tags": bio_tags(schema),
              "entities": list(schema["entities"]), "relations": list(schema["relations"])}
    json.dump(report, open("outputs/reports/pilot_report.json", "w"), indent=2)

    print(f"Random pilot: {len(sample)} records (from {dstats['unique_exact_pairs']} unique pairs)")
    print("composition (report only):", audit["pilot_composition_random"])
    print("pre-annotation coverage :", report["preannotation_coverage"])
    print("\nImport into Label Studio:")
    print("  * project labeling config -> outputs/labeling_config.xml")
    print("  * tasks (with pre-annotations) -> outputs/pilot/pilot_tasks.json")


if __name__ == "__main__":
    main()

