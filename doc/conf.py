# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

from datetime import date

import sphinx_gallery
import sphinx_bootstrap_theme
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder

sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'ndspflow'
copyright = '2020-{}, VoytekLab'.format(date.today().year)
author = 'Ryan Hammonds'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_gallery.load_style',
    'sphinx_copybutton',
    'nbsphinx',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# numpydoc interacts with autosummary, that creates excessive warnings
# This line is a 'hack' for that interaction that stops the warnings
numpydoc_show_class_members = False

# Set to generate sphinx docs for class members (methods)
autodoc_default_options = {
    'members': None,
    'inherited-members': None,
}

# generate autosummary even if no references
autosummary_generate = True

# The suffix(es) of source filenames. Can be str or list of string
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Settings for sphinx_copybutton
copybutton_prompt_text = "$ "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'bootstrap'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Set the theme path explicitly
#   This isn't always needed, but is useful so bulding docs doesn't fail on
#   operating systems which don't have bootstrap on theme path
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options to customize the look and feel, which are theme-specific.
html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("API", "api"),
        ("Tutorials", "tutorials/index"),
        ("GitHub", "https://github.com/voytekresearch/ndspflow", True),
    ],

    # Set the page width to not be restricted to hardset value
    'body_max_width': None,

    # Bootswatch (http://bootswatch.com/) theme to apply.
    'bootswatch_theme': "flatly",

    # Render the current pages TOC in the navbar
    'navbar_pagenav': False,
}

# Settings for whether to copy over and show link rst source pages
html_copy_source = False
html_show_sourcelink = False

# -- Extension configuration -------------------------------------------------

# Configurations for sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['tutorials'],
    'gallery_dirs': ['auto_tutorials'],
    'within_subsection_order': FileNameSortKey,
    'backreferences_dir': 'generated',   # Where to drop linking files between examples & API
    'doc_module': ('ndspflow',),
    'reference_url': {'ndspflow': None},
    'remove_config_comments': True,
}
