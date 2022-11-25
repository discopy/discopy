# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath("./_ext"))


def get_version():
    from discopy import __version__
    return __version__


# -- Project information -----------------------------------------------------

project = 'DisCoPy'
copyright = '2019, DisCoPy'
author = 'oxford-quantum-group'

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'm2r2',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'youtube',
              'bases-fullname',
              ]

autosummary_generate = True

autodoc_mock_imports = ["pytket", "pennylane", "torch"]

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = False

napoleon_use_admonition_for_examples = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['imgs']

html_title = "DisCoPy"

html_theme_options = {
    "repository_url": "https://github.com/oxford-quantum-group/discopy",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "extra_navbar": "",
}

master_doc = 'index'
