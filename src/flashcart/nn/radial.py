import math
from typing import Literal, Optional

import torch
import torch.nn as nn


class BesselBasis(nn.Module):
    """Evaluate a radial basis proportional to spherical Bessel functions.

    For ``n = 1, ..., n_radial``, the basis is
    ``sqrt(2 / r_max) * sin(n * pi * r / r_max) / r``. Evaluation uses the continuous
    limit at zero. The basis is not masked beyond ``r_max``. A separate cutoff provides
    that restriction.

    Args:
        n_radial (int): Number of basis functions.
        r_max (float): Cutoff radius.
    """

    def __init__(
        self,
        n_radial: int,
        r_max: float,
    ):
        super().__init__()
        if n_radial <= 0:
            raise ValueError(f"n_radial must be positive. Provided: {n_radial}.")
        if r_max <= 0.0:
            raise ValueError(f"r_max must be positive. Provided: {r_max}.")

        dtype = torch.get_default_dtype()

        self.n_radial = n_radial

        weights = (math.pi / r_max) * torch.arange(1, n_radial + 1, dtype=dtype)

        self.register_buffer("bessel_weights", weights)
        assert self.bessel_weights.numel() == self.n_radial

        self.register_buffer("r_cutoff", torch.tensor(r_max, dtype=dtype))
        self.register_buffer("pre_factor", torch.tensor(math.sqrt(2.0 / r_max), dtype=dtype))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis.

        Args:
            distances (torch.Tensor): Edge lengths, shape ``(n_edges,)``.

        Returns:
            torch.Tensor: Basis values, shape ``(n_edges, n_radial)``.
        """
        x = distances.unsqueeze(-1) * self.bessel_weights
        return self.pre_factor * self.bessel_weights * torch.sinc(x / math.pi)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"r_cutoff={float(self.r_cutoff)}, " f"n_radial={self.n_radial})"


class ChebyshevBasis(nn.Module):
    """Evaluate Chebyshev polynomials T_0..T_{n-1} on mapped distances.

    Distances in ``[r_min, r_max]`` are mapped to ``t`` in ``[0, 1]``. The
    Chebyshev input is ``2*t - 1`` for the affine mapping or ``cos(pi*t)`` for
    the cosine mapping. The polynomials are evaluated with the three-term
    recurrence.

    Distances outside the mapping interval are evaluated without clipping. The cutoff
    envelope is applied separately.

    Args:
        n_radial (int): Number of basis functions.
        r_max (float): Upper bound of the mapping interval (cutoff radius).
        r_min (float, optional): Lower bound of the mapping interval. Default: 0.0.
        mapping (str, optional): "affine" or "cosine" distance mapping. Default:
            "affine".
    """

    def __init__(
        self,
        n_radial: int,
        r_max: float,
        r_min: float = 0.0,
        mapping: Literal["cosine", "affine"] = "affine",
    ):
        super().__init__()
        if r_max <= r_min:
            raise ValueError("Expected r_max > r_min. " f"Provided: r_min={r_min}, r_max={r_max}.")
        if mapping not in ("cosine", "affine"):
            raise ValueError("Expected mapping to be 'cosine' or 'affine'. " f"Provided: {mapping}.")

        dtype = torch.get_default_dtype()
        self.n_radial = n_radial
        self.mapping = mapping
        self.register_buffer("r_min", torch.tensor(r_min, dtype=dtype))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=dtype))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis.

        Args:
            distances (torch.Tensor): Edge lengths, shape ``(n_edges,)``.

        Returns:
            torch.Tensor: Basis values, shape ``(n_edges, n_radial)``.
        """
        distances_norm = (distances - self.r_min) / (self.r_max - self.r_min)
        if self.mapping == "cosine":
            x = torch.cos(distances_norm * math.pi)
        else:
            x = 2.0 * distances_norm - 1.0

        if self.n_radial == 1:
            basis = torch.ones_like(x).unsqueeze(-1)
        else:
            T = [torch.ones_like(x), x]
            for _ in range(2, self.n_radial):
                T.append(2.0 * x * T[-1] - T[-2])
            basis = torch.stack(T, dim=-1)

        return basis

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_radial={self.n_radial}, r_min={float(self.r_min):.4g}, "
            f"r_max={float(self.r_max):.4g}, mapping={self.mapping})"
        )


class PolynomialCutoff(nn.Module):
    """Polynomial envelope with continuous first and second derivatives at the cutoff.

    For ``t = r / r_max``, the envelope is ``1 - c1*t**p + c2*t**(p+1) - c3*t**(p+2)``.
    The coefficients make the value and its first two derivatives vanish at ``r_max``.
    Negative distances are treated as zero, and values at or beyond the cutoff are set
    to zero.

    Args:
        r_max (float): Cutoff radius.
        poly_order (int, optional): Polynomial order p (>= 2). Default: 6.
    """

    def __init__(self, r_max: float, poly_order: int = 6):
        super().__init__()
        dtype = torch.get_default_dtype()

        if poly_order < 2:
            raise ValueError("Polynomial order must be >= 2. " f"Provided: {poly_order}.")

        self.poly_order_value = int(poly_order)
        self.register_buffer("poly_order", torch.tensor(poly_order, dtype=torch.long))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=dtype))

        self.register_buffer(
            "coeff1",
            torch.tensor((poly_order + 1.0) * (poly_order + 2.0) / 2.0, dtype=dtype),
        )
        self.register_buffer("coeff2", torch.tensor(poly_order * (poly_order + 2.0), dtype=dtype))
        self.register_buffer("coeff3", torch.tensor(poly_order * (poly_order + 1.0) / 2.0, dtype=dtype))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate the cutoff envelope (zero at and beyond ``r_max``).

        Args:
            distances (torch.Tensor): Edge lengths, shape ``(n_edges,)``.

        Returns:
            torch.Tensor: Envelope values with shape ``(n_edges,)``.
        """
        return self.calculate_envelope(
            distances,
            self.r_max,
            self.poly_order_value,
            self.coeff1,
            self.coeff2,
            self.coeff3,
        )

    @staticmethod
    def calculate_envelope(
        distances: torch.Tensor,
        r_max: torch.Tensor,
        poly_order: int,
        coeff1: Optional[torch.Tensor] = None,
        coeff2: Optional[torch.Tensor] = None,
        coeff3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the envelope without a module instance.

        Args:
            distances (torch.Tensor): Distances, clamped to >= 0.
            r_max (torch.Tensor): Cutoff radius.
            poly_order (int): Polynomial order p.
            coeff1 (torch.Tensor, optional): Precomputed ``(p+1)(p+2)/2``. Default:
                None, computed from ``poly_order``.
            coeff2 (torch.Tensor, optional): Precomputed ``p(p+2)``. Default: None,
                computed from ``poly_order``.
            coeff3 (torch.Tensor, optional): Precomputed ``p(p+1)/2``. Default: None,
                computed from ``poly_order``.

        Returns:
            torch.Tensor: Envelope values with the same shape as ``distances``.
        """
        distances = torch.clamp(distances, min=0.0)
        distances_norm = distances / r_max

        if poly_order <= 4:
            pow_p = torch.pow(distances_norm, poly_order)
            pow_p1 = torch.pow(distances_norm, poly_order + 1)
            pow_p2 = torch.pow(distances_norm, poly_order + 2)
        else:
            pow_p = distances_norm.clone()
            for _ in range(poly_order - 1):
                pow_p = pow_p * distances_norm

            pow_p1 = pow_p * distances_norm
            pow_p2 = pow_p1 * distances_norm

        if coeff1 is None:
            coeff1 = torch.tensor(
                (poly_order + 1.0) * (poly_order + 2.0) / 2.0,
                dtype=distances.dtype,
                device=distances.device,
            )
        if coeff2 is None:
            coeff2 = torch.tensor(
                poly_order * (poly_order + 2.0),
                dtype=distances.dtype,
                device=distances.device,
            )
        if coeff3 is None:
            coeff3 = torch.tensor(
                poly_order * (poly_order + 1.0) / 2.0,
                dtype=distances.dtype,
                device=distances.device,
            )

        envelope = 1.0 - coeff1 * pow_p + coeff2 * pow_p1 - coeff3 * pow_p2

        mask = (distances < r_max).to(dtype=envelope.dtype)
        return torch.clamp(envelope * mask, min=0.0, max=1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_max={float(self.r_max):.4g}, " f"poly_order={self.poly_order.item()})"
