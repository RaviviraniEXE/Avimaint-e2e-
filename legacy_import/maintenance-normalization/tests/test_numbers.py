from src.utils.numbers import digitize, spell


def test_digitize_simple():
    assert digitize("number two")[0] == "number 2"
    assert digitize("one hundred and fifty degrees")[0] == "150 degrees"


def test_digitize_list_not_summed():
    assert digitize("numbers one, two and three")[0] == "numbers 1, 2 and 3"


def test_digitize_range():
    assert digitize("one hundred to one hundred and fifty")[0] == "100 to 150"


def test_digitize_scale():
    assert digitize("twelve hundred rpm")[0] == "1200 rpm"


def test_digitize_no_flag_on_plain_text():
    out, flag = digitize("engine idle killed engine")
    assert flag is False and out == "engine idle killed engine"


def test_spell_roundtrip_ish():
    assert spell("number 2") == "number two"
    assert spell("150 degrees") == "one hundred and fifty degrees"

