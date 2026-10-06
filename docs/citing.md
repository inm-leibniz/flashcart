# Citing FlashCart

If you use FlashCart in your work, please cite the
{ref}`FlashCart paper <flashcart-paper>`:

```bibtex
@misc{zaverkin2026,
      title={FlashCart: Fast Cartesian Tensor Products for Equivariant Interatomic Potentials},
      author={Viktor Zaverkin and Payman Goodarzi and Sergey V. Sukhomlinov and Davit Hovhannisyan and Roland Aydin and Martin H. Müser and Mathias Niepert},
      year={2026},
      eprint={2610.06409},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.06409},
}
```

See {doc}`publications` for the package version, configurations, and examples
accompanying the {ref}`FlashCart paper <flashcart-paper>`.

FlashCart builds on the framework for equivariant message passing with
irreducible Cartesian tensors introduced in ICTP:

Zaverkin et al., [Higher-Rank Irreducible Cartesian Tensors for Equivariant Message
Passing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e00573ccc366b57d0d66497c2e21919f-Abstract-Conference.html),
*Advances in Neural Information Processing Systems* **37** (NeurIPS 2024).

## License

FlashCart is released under the Apache License 2.0. It includes code adapted
from PyTorch Geometric and pytorch_scatter (MIT) and PyTorch (BSD 3-Clause).
See the `NOTICE` file.
