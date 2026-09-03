from src.data.dictionary import build_lexicon
from src.data.load import load_config

cfg = load_config()
lex, rep = build_lexicon(cfg)


def test_single_abbrev_accepted():
    assert lex.abbrev.get("cyl") == "cylinder"
    assert lex.abbrev.get("eng") == "engine"
    assert lex.abbrev.get("r/h") == "right-hand"


def test_ambiguous_abbrev_skipped():
    # COMP has two expansions (compression/compressor) -> skipped
    assert "comp" in lex.ambiguous
    assert "comp" not in lex.abbrev


def test_keep_list_protected():
    for k in ("cht", "egt", "psi", "rpm", "fod"):
        assert k in lex.keep


def test_symbols_present():
    assert lex.symbols.get("#") == "number"
    assert lex.symbols.get("&") == "and"
    assert lex.symbols.get("*") == "degrees"


def test_risky_misspelling_skipped():
    # 'OFF'->'OF' would corrupt valid text -> must be skipped
    assert "off" not in lex.misspellings
    # a genuine misspelling is kept
    assert lex.misspellings.get("engien") == "engine"

