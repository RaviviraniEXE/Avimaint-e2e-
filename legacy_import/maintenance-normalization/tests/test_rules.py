from src.normalization.system_b_rules import RuleBasedNormalizer

norm = RuleBasedNormalizer.from_config()


def n(t):
    return norm.normalize(t).normalized


def test_number_and_symbol():
    assert n("#2 & 4 CYL ROCKER COVER GASKETS ARE LEAKING.") == \
        "number 2 and 4 cylinder rocker cover gaskets are leaking."


def test_slash_abbrev():
    assert n("R/H MAG BLAST TUBE FELL OFF.") == "right-hand magneto blast tube fell off."


def test_alt_expands_single():
    assert n("R/H ENG FWD ALT ATTACH BOLT LOOSE.") == \
        "right-hand engine forward alternator attach bolt loose."


def test_misspelling_fixed():
    assert "engine" in n("ENGIEN WOULD NOT START")
    assert "engien" not in n("ENGIEN WOULD NOT START")


def test_keep_as_is():
    assert "psi" in n("30 PSI")
    assert "cht" in n("CHT PROBE")          # kept, not expanded


def test_ambiguous_not_expanded():
    assert "comp" in n("LOW COMP ON CYL")   # COMP ambiguous -> left alone


def test_degrees_symbol():
    assert n("TEMP 150*F") == "temperature 150 degrees fahrenheit"


def test_alignment_present():
    res = norm.normalize("R/H MAG")
    assert res.alignment and res.alignment[0][0] == "R/H"

