# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'evopt'
copyright = '2025, Roberto Hart-Villamil'
author = 'Roberto Hart-Villamil'
release = '2024'



# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Handle import errors for autodoc generation -----------------------------
try:
    import evopt
except ImportError:
    # If the package isn't installed, use autodoc_mock_imports
    autodoc_mock_imports = [
    'numpy', 
    'pandas', 
    'matplotlib', 
    'cma',
    'cloudpickle',
    'scipy',
    'plotly'
]

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgconverter',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
]
autosectionlabel_prefix_document = True
suppress_warnings = [
    'app.add_directive', # Suppress warnings about directives being added twice
    'app.duplicated_objectid',
    'app.add_node', 
    'app.add_role',
    'app.add_generic_role',
    'app.add_source_parser',
    'autosectionlabel.*',  # Suppress warnings about duplicate section labels
    'image.nonlocal_uri',
    'toc.secnum',  # Table of contents section numbering warnings
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
