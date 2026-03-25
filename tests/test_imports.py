"""Smoke tests for core imports."""


def test_core_imports() -> None:
    import splatcredb
    from splatcredb import baselines, features, metrics, oracle, pruning, render, report
    from tools import evaluate, export_report, prune_and_render

    assert splatcredb.__version__ == "0.1.0"
    assert all([baselines, features, metrics, oracle, pruning, render, report, evaluate, export_report, prune_and_render])
