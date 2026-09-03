from src.evaluation.metrics import extrinsic_report, intrinsic_report, oov_rate


def test_oov_rate():
    assert oov_rate(["engine cylinder"], {"engine", "cylinder"}) == 0.0
    assert oov_rate(["engine widget"], {"engine", "cylinder"}) == 0.5


def test_intrinsic_reduces_oov():
    rep = intrinsic_report(["cyl eng"], ["cylinder engine"],
                           [{"n_expansions": 2, "n_tokens": 2}], {"cylinder", "engine"})
    assert rep["norm_oov"] <= rep["raw_oov"]
    assert rep["expansions"] == 2


def test_extrinsic_err():
    rep = extrinsic_report(["cylinder number 4 intake leaking."],
                           ["cylinder number 4 intake leaking."],
                           ["cyl #4 intake leaking."])
    assert rep["exact_match"] == 1.0 and rep["err_word"] == 1.0

