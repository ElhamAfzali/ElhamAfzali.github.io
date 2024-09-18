# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath('../batorch'))
# sys.path.insert(0, os.path.abspath('..'))
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)
sys.path.insert(0, os.path.join(basedir, "kernax"))
print(sys.path)

import kernax
project = 'Kernax'
copyright = 'Safran Group'
author = 'Safran Group'

release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx.ext.intersphinx",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "myst_nb",
]

numfig = True
autosummary_generate = True

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_css_files = ['custom.css']
html_static_path = ["_static"]
html_title = "Kernax"
html_css_files = ["custom.css"]
html_context = {
   "default_mode": "light"
}

autodoc_mock_imports = ["jax"]  #"jax.numpy", "kernax", "kernax.thinning", "kernax.discrepancies", "kernax.kernels"]
autodoc_member_order = 'bysource'