import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--diagnosticsresample", action="store_true"
    )


@pytest.fixture
def diagnostics_option(request):
    return request.config.getoption("--diagnosticsresample")
