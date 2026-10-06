"""Sphinx configuration for the FlashCart documentation."""

import os
import tempfile

# Use a cache in the temporary directory unless one is already configured.
os.environ.setdefault(
    "FLASHCART_CACHE_DIR", os.path.join(tempfile.gettempdir(), "flashcart-docs-cache")
)

import flashcart  # noqa: E402

project = "FlashCart"
author = "The FlashCart Authors"
copyright = "2026, The FlashCart Authors"
release = flashcart.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# Google-style docstrings; constructor arguments are documented on the class.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "show-inheritance": True}
# Optional runtime dependencies that are not installed on the docs builder.
autodoc_mock_imports = ["lammps", "cupy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "ase": ("https://docs.ase-lib.org", None),
}

html_theme = "furo"
html_title = f"FlashCart {release}"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["branding.css"]
html_favicon = "_static/flashcart-favicon.png"

# Parameter-group hooks used internally by the optimizer routing.
_HIDDEN_MEMBERS = {"non_decayable_parameters", "non_muon_parameters"}


def _skip_member(app, what, name, obj, skip, options):
    return True if name in _HIDDEN_MEMBERS else None


def setup(app):
    app.connect("autodoc-skip-member", _skip_member)
