def test_package_exposes_version():
    import cfm_grn_v2
    assert isinstance(cfm_grn_v2.__version__, str)
