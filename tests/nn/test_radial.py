import numpy as np
import pytest
import torch

from flashcart.nn.radial import ChebyshevBasis, PolynomialCutoff


@pytest.mark.parametrize("poly_order", [3, 4, 5, 6])
def test_polynomial_cutoff_matches_closed_form(poly_order: int) -> None:
    r_max = 4.0
    cutoff = PolynomialCutoff(r_max=r_max, poly_order=poly_order).to(torch.float64)
    r = torch.cat(
        [
            torch.linspace(0.0, r_max - 1.0e-6, 101, dtype=torch.float64),
            torch.tensor([r_max, r_max + 0.5, 2.0 * r_max], dtype=torch.float64),
        ]
    )
    envelope = cutoff(r)

    p = float(poly_order)
    x = (r / r_max).numpy()
    poly = 1.0 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
    expected = np.clip(poly * (r.numpy() < r_max), 0.0, 1.0)
    torch.testing.assert_close(envelope, torch.from_numpy(expected), atol=1.0e-12, rtol=1.0e-12)

    assert envelope[0] == 1.0
    assert torch.all(envelope[-3:] == 0.0)
    assert envelope[100] < 1.0e-12
    assert torch.all(envelope[1:101] <= envelope[:100] + 1.0e-12)


@pytest.mark.parametrize("mapping", ["affine", "cosine"])
@pytest.mark.parametrize("n_radial", [1, 5])
def test_chebyshev_basis_matches_numpy(mapping: str, n_radial: int) -> None:
    r_min, r_max = 0.5, 4.0
    basis = ChebyshevBasis(n_radial=n_radial, r_max=r_max, r_min=r_min, mapping=mapping).to(torch.float64)
    r = torch.linspace(r_min, r_max, 33, dtype=torch.float64)
    out = basis(r)

    norm = ((r - r_min) / (r_max - r_min)).numpy()
    x = np.cos(norm * np.pi) if mapping == "cosine" else 2.0 * norm - 1.0
    expected = np.polynomial.chebyshev.chebvander(x, n_radial - 1)
    torch.testing.assert_close(out, torch.from_numpy(expected), atol=1.0e-12, rtol=1.0e-12)
