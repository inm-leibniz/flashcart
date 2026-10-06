import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"

import pytest
import torch


@pytest.fixture(scope="module")
def fp64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
