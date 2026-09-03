"""Step 2 — build + audit the lexicon from Amin's expert lists."""
import _bootstrap  # noqa: F401
import os

from src.data.dictionary import build_lexicon
from src.data.load import load_config


def main():
    cfg = load_config()
    lex, rep = build_lexicon(cfg)
    print(f"abbreviations accepted : {len(lex.abbrev)}")
    print(f"abbreviations ambiguous: {len(lex.ambiguous)}")
    print(f"symbols                : {len(lex.symbols)}")
    print(f"misspellings (single)  : {len(lex.misspellings)}")
    print(f"misspellings (multi)   : {len(lex.misspell_multi)}")
    print(f"keep-as-is             : {len(lex.keep)}")
    d = cfg["evaluation"]["report_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "dictionary_build_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep.to_text())
    print(f"\nreport -> {os.path.join(d, 'dictionary_build_report.txt')}")


if __name__ == "__main__":
    main()

