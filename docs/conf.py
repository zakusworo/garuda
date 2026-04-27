# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

from garuda._version import __version__

# -- Project information -----------------------------------------------------
project = "GARUDA"
copyright = "2025, Zulfikar Aji Kusworo"
author = "Zulfikar Aji Kusworo"
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}

# -- Autodoc settings ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Numpydoc settings -------------------------------------------------------
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# -- Coverage / nitpicky -----------------------------------------------------
nitpicky = False
