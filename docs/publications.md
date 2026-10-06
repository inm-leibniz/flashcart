# Publications

Publications using FlashCart are listed below, together with the package
version and resources accompanying each study.

(flashcart-paper)=
## FlashCart: Fast Cartesian Tensor Products for Equivariant Interatomic Potentials

V. Zaverkin, P. Goodarzi, S. V. Sukhomlinov, D. Hovhannisyan, R. Aydin,
M. H. Müser, and M. Niepert, [arXiv:2610.06409](https://arxiv.org/abs/2610.06409) (2026).
See {doc}`citing` for the BibTeX entry.

- **FlashCart version:** 0.1.0.
- **Trained models:** [Zenodo](https://doi.org/10.5281/zenodo.23159584).
- **Configurations:** `examples/configs/make_configs.py` generates configurations
  for the five SPICE models, each with seeds 0, 1, and 2.
  `examples/configs/mad.yaml` defines the MAD model, also trained with
  seeds 0, 1, and 2. See {doc}`tutorials/configs` for training and evaluation
  commands.
- **Data preparation:** {doc}`tutorials/datasets` describes how to download and
  prepare the SPICE and MAD datasets.
- **Molecular dynamics:** {doc}`tutorials/md` demonstrates simulations of liquid
  water using the inputs in `examples/md/water`.
- **Runtime and memory measurements:** FlashCart measurements use an NVIDIA RTX
  PRO 6000 Blackwell Server Edition, PyTorch 2.11.0, CUDA 12.8, and float32
  with TF32 disabled. See {doc}`tutorials/timing` for an example of the model
  timing procedure.

Install the release used in this study with:

```bash
python -m pip install flashcart==0.1.0
```

To add a publication that uses FlashCart, open a pull request with its citation,
package version, and links to the relevant configurations or data.
