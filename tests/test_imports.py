def test_core_imports() -> None:
    import splatcredb
    import splatcredb.baselines
    import splatcredb.features
    import splatcredb.metrics
    import tools.evaluate
    assert splatcredb.__version__ == "0.1.0"
