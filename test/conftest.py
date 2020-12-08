def pytest_addoption(parser):

    # Used in test_resample_spatial
    parser.addoption(
        "--diagnosticsresample", action="store_true"
    )
