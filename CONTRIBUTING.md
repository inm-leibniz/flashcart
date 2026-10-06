# Contributing

Bug reports and pull requests are welcome.

- **Issues:** include the FlashCart, PyTorch, and CUDA versions, the GPU model, and a
  minimal configuration or script that reproduces the problem.
- **Development installation:** run `pip install -e ".[dev,docs]"` from the cloned repository root.
- **Tests:** `pytest` runs the default suite. GPU tests are skipped when CUDA is
  unavailable. `pytest -m slow` runs the tests marked as slow, including CLI and
  multiprocess tests. Test kernel changes on a CUDA GPU as well.
- **Code style:** format Python code with `black`, using the configured line length
  of 120 columns.
- **Documentation:** use Google-style docstrings and document constructor arguments
  in the class docstring. Follow the formatting of the surrounding documentation.
  Run `sphinx-build -W --keep-going -b html docs docs/_build/html`.
  The documentation must build without warnings.
- **Changelog:** add an entry under "Unreleased" in `CHANGELOG.md` for each
  user-visible change.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
