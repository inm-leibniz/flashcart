import math

import numpy as np
import pytest

from flashcart.data.utils import numbers_to_types, types_to_one_hot


def _make_config(atomic_numbers, positions, energy=None, forces=None):
    from flashcart.data.data import AtomicConfig

    config = AtomicConfig(
        atomic_numbers=np.array(atomic_numbers, dtype=np.int64),
        positions=np.array(positions, dtype=np.float64),
        energy=float(energy) if energy is not None else None,
        forces=np.array(forces, dtype=np.float64) if forces is not None else None,
    )
    config.compute_neighbors(r_max=3.0)
    return config


def _make_loader(configs, elements):
    from flashcart.data.dataset import AtomicDataset
    from flashcart.utils.torch_geometric.dataloader import DataLoader

    return DataLoader(AtomicDataset(configs, r_max=3.0, elements=elements), batch_size=8)


def test_numbers_to_types():
    z = np.array([1, 6, 8, 1], dtype=np.int64)
    assert np.allclose(numbers_to_types(z, ["H", "C", "O"]), [0, 1, 2, 0])

    with pytest.raises(ValueError, match="atomic numbers not in elements"):
        numbers_to_types(np.array([1, 92], dtype=np.int64), ["H", "C", "O"])

    with pytest.raises(ValueError, match="Duplicate"):
        numbers_to_types(np.array([1], dtype=np.int64), ["H", "H"])


def test_types_to_one_hot():
    oh = types_to_one_hot(np.array([0, 2, 1], dtype=np.int64), n_elements=3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    assert np.allclose(oh, expected)


def test_atomic_energy_shifts():
    from flashcart.data.statistics import get_atomic_energy_shifts

    configs = [
        _make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]], energy=-20.0),
        _make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]], energy=-30.0),
    ]
    shifts = get_atomic_energy_shifts(_make_loader(configs, ["H"]), n_elements=1)
    assert abs(shifts[0] - (-12.5)) < 0.01

    configs = [
        _make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]], energy=-10.0),
        _make_config([6], [[0, 0, 0]], energy=-10.0),
        _make_config([1, 6], [[0, 0, 0], [1.4, 0, 0]], energy=-15.0),
    ]
    shifts = get_atomic_energy_shifts(_make_loader(configs, ["H", "C"]), n_elements=2, lam=1e-6)
    assert abs(shifts[0] - (-5.0)) < 0.1
    assert abs(shifts[1] - (-10.0)) < 0.1

    shifts = get_atomic_energy_shifts(
        _make_loader(configs, ["H", "C"]),
        n_elements=2,
        atomic_shifts=np.array([1.5, -2.5]),
        fit=False,
    )
    assert np.allclose(shifts, [1.5, -2.5])

    with pytest.raises(ValueError, match="atomic_shifts must be provided"):
        get_atomic_energy_shifts(_make_loader(configs, ["H", "C"]), n_elements=2, fit=False)

    with pytest.raises(ValueError, match="atomic_shifts must have shape"):
        get_atomic_energy_shifts(
            _make_loader([], ["H", "C"]),
            n_elements=2,
            atomic_shifts=np.array([-5.0]),
        )


def test_force_rms():
    from flashcart.data.statistics import get_force_rms

    configs = [_make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]], energy=-20.0, forces=[[1, 0, 0], [1, 0, 0]])]
    rms = get_force_rms(_make_loader(configs, ["H"]))
    assert abs(rms - math.sqrt(2 / 6)) < 1e-6

    configs = [_make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]], energy=-20.0, forces=[[0, 0, 0], [0, 0, 0]])]
    result = get_force_rms(_make_loader(configs, ["H"]), n_elements=1)
    assert result[0] == 1.0


def test_average_n_neighbors():
    from flashcart.data.statistics import get_avg_neighbors

    configs = [_make_config([1, 1], [[0, 0, 0], [1.4, 0, 0]])]
    assert abs(get_avg_neighbors(_make_loader(configs, ["H"])) - 1.0) < 1e-6

    h = 1.5 * math.sqrt(3) / 2
    positions = [[0, 0, 0], [1.5, 0, 0], [0.75, h, 0]]
    configs = [_make_config([1, 1, 1], positions)]
    assert abs(get_avg_neighbors(_make_loader(configs, ["H"])) - 2.0) < 1e-6

    configs = [_make_config([1, 1, 1], [[0, 0, 0], [1.0, 0, 0], [50.0, 0, 0]])]
    assert abs(get_avg_neighbors(_make_loader(configs, ["H"])) - 2.0 / 3.0) < 1e-6
