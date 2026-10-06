import pytest
import torch

from helpers import random_rotation
from flashcart.o3.utils import rotate_irreps

pytestmark = pytest.mark.usefixtures("fp64_default")

L_MAX = 3


def test_rotate_irreps_matches_composed_rotation():
    torch.manual_seed(1234)
    x = torch.randn(8, (L_MAX + 1) ** 2, dtype=torch.float64)
    r1 = random_rotation(seed=1, dtype=torch.float64)
    r2 = random_rotation(seed=2, dtype=torch.float64)

    sequential = rotate_irreps(rotate_irreps(x, r1, L_MAX), r2, L_MAX)
    composed = rotate_irreps(x, r2 @ r1, L_MAX)
    torch.testing.assert_close(sequential, composed, atol=1.0e-12, rtol=1.0e-12)


def test_rotate_irreps_l1_matches_cartesian():
    torch.manual_seed(1234)
    x = torch.randn(8, (L_MAX + 1) ** 2, dtype=torch.float64)
    rotation = random_rotation(seed=3, dtype=torch.float64)

    rotated = rotate_irreps(x, rotation, L_MAX)
    torch.testing.assert_close(rotated[:, 1:4], x[:, 1:4] @ rotation.T, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(rotated[:, :1], x[:, :1], atol=0.0, rtol=0.0)
