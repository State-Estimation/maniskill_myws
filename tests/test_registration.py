def test_register_import():
    # This test intentionally does not require ManiSkill to be importable.
    # It only checks that our package-level register() exists.
    import maniskill_myws

    assert hasattr(maniskill_myws, "register")


