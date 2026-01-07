import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)


def pytest_runtest_setup(item):
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
