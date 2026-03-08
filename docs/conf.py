"""Sphinx configuration for Chronax documentation."""

project = "Chronax"
author = "Simulacrum, Inc"
copyright = "2025, Simulacrum, Inc"

version = "0.1"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = "bysource"

# HTML output
html_theme = "sphinx_rtd_theme"
