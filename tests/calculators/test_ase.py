import numpy as np
import pytest
from ase.build import bulk, molecule

from helpers import tiny_model_cfg
from flashcart.calculators import FlashCartCalculator
from flashcart.model.flashcart import FlashCartPotential


def _checkpoint(tmp_path):
    cfg = tiny_model_cfg(
        elements=["C", "H", "O"],
        n_hidden_feats=4,
        l_max_hidden_feats=1,
        l_max_edge_attrs=2,
        n_radial=4,
        correlation=2,
        hidden_radial=[8, 8],
        hidden_readout=[8, 8],
        r_max=4.0,
        fit_atomic_shifts=False,
        avg_neighbors=20.0,
    )
    model = FlashCartPotential.from_config(cfg)
    ckpt = tmp_path / "best"
    model.save_inference_checkpoint(ckpt, meta={"epoch": 0})
    return ckpt


def test_padded_calculator_matches_unpadded(tmp_path):
    ckpt = _checkpoint(tmp_path)
    atoms = bulk("C", "diamond", a=3.57, cubic=True)

    plain = FlashCartCalculator.from_checkpoint(ckpt, device="cpu")
    padded = FlashCartCalculator.from_checkpoint(ckpt, device="cpu", pad_n_atoms=64, pad_n_edges=512)

    atoms.calc = plain
    e_plain = atoms.get_potential_energy()
    f_plain = atoms.get_forces()
    s_plain = atoms.get_stress()

    atoms.calc = padded
    assert np.allclose(atoms.get_potential_energy(), e_plain, atol=1.0e-6, rtol=1.0e-6)
    assert np.allclose(atoms.get_forces(), f_plain, atol=1.0e-6, rtol=1.0e-6)
    assert np.allclose(atoms.get_stress(), s_plain, atol=1.0e-6, rtol=1.0e-6)


def test_calculator_independent_of_neighbor_skin(tmp_path):
    ckpt = _checkpoint(tmp_path)
    atoms = molecule("CH3CH2OH")

    without = FlashCartCalculator.from_checkpoint(ckpt, device="cpu", skin=0.0)
    with_skin = FlashCartCalculator.from_checkpoint(ckpt, device="cpu", skin=2.0)

    atoms.calc = without
    e_ref = atoms.get_potential_energy()
    f_ref = atoms.get_forces()

    atoms.calc = with_skin
    assert np.allclose(atoms.get_potential_energy(), e_ref, atol=1.0e-5, rtol=1.0e-5)
    assert np.allclose(atoms.get_forces(), f_ref, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.slow
def test_compiled_calculator_matches_eager(tmp_path):
    ckpt = _checkpoint(tmp_path)
    atoms = molecule("CH3CH2OH")

    eager = FlashCartCalculator.from_checkpoint(ckpt, device="cpu")
    atoms.calc = eager
    e_eager = atoms.get_potential_energy()
    f_eager = atoms.get_forces()

    compiled = FlashCartCalculator.from_checkpoint(ckpt, device="cpu", compile_mode="default", compile_fullgraph=True)
    atoms.calc = compiled
    e_compiled = atoms.get_potential_energy()
    f_compiled = atoms.get_forces()

    assert np.allclose(e_eager, e_compiled, atol=1.0e-5, rtol=1.0e-4)
    assert np.allclose(f_eager, f_compiled, atol=1.0e-5, rtol=1.0e-4)
