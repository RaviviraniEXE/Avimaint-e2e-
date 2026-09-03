"""Step 1 — build the combined records and the gold from Amin's cleaned data.

Produces:
  data/interim/records_combined.csv   IDENT, RAW   (PROBLEM + ACTION combined)
  data/processed/gold_normalization.csv  IDENT, RAW, GOLD [, digitize_flagged]

Gold = Amin's expert-cleaned text, lowercased, with numbers converted to digits
when config numbers == 'digits' (recommended for IE). In 'words' mode the gold
is used exactly as Amin published it.
"""
import _bootstrap  # noqa: F401
import os

import pandas as pd

from src.data.load import combined_text, load_config, load_expert_cleaned, load_logbook
from src.utils.numbers import digitize


def main():
    cfg = load_config()
    log = load_logbook(cfg)
    lc = cfg["logbook"]

    # combined raw records
    recs = [{"IDENT": r[lc["id_col"]],
             "RAW": combined_text(r[lc["problem_col"]], r[lc["action_col"]])}
            for _, r in log.iterrows()]
    rec_df = pd.DataFrame(recs)
    os.makedirs(cfg["paths"]["interim_dir"], exist_ok=True)
    rec_df.to_csv(os.path.join(cfg["paths"]["interim_dir"], "records_combined.csv"), index=False)
    print(f"Combined records: {len(rec_df)}")

    # gold from Amin cleaned
    cl = load_expert_cleaned(cfg)
    mode = cfg.get("numbers", "digits")
    raw_by_id = dict(zip(rec_df["IDENT"], rec_df["RAW"]))
    rows, flagged = [], 0
    for _, r in cl.iterrows():
        gold = r["CLEANED"]
        flag = False
        if mode == "digits":
            gold, flag = digitize(gold)
            flagged += int(flag)
        rows.append({"IDENT": r["ID"], "RAW": raw_by_id.get(r["ID"], ""),
                     "GOLD": gold.lower(), "digitize_flagged": str(flag)})
    gold_df = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    gold_df.to_csv(cfg["evaluation"]["gold_file"], index=False)
    print(f"Gold ({mode}): {len(gold_df)} rows -> {cfg['evaluation']['gold_file']}")
    if mode == "digits":
        print(f"  number-conversion flagged (needs human check): {flagged}")


if __name__ == "__main__":
    main()

