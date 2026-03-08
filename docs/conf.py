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
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = "bysource"

# Intersphinx: cross-reference JAX, NumPy, and Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
}

# Copybutton: strip common prompts from copied code
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# HTML output
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/Smlcrm/ml-library-chronax",
    "show_prev_next": False,
    "navbar_align": "left",
}
