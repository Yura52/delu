import datetime
import platform
import sys
from pathlib import Path

# add the repository root to PYTHONPATH
delu_path = Path.cwd()
while not (delu_path.name == 'delu' and delu_path.parent.name != 'delu'):
    delu_path = delu_path.parent
sys.path.append(str(delu_path))
import delu  # noqa

# >>> project >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
author = 'Yura52'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)
project = 'DeLU'
release = delu.__version__
version = delu.__version__

# >>> general >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
default_role = 'py:obj'
pygments_style = 'default'
repo_url = 'https://github.com/Yura52/delu'
templates_path = ['_templates']

# >>> extensions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]
if platform.machine() != 'arm64':
    # libenchant is not available for Apple CPUs
    extensions.append('sphinxcontrib.spelling')

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

doctest_global_setup = '''
import numpy as np
import torch
import delu
from delu import *
from delu.data import *
from delu.hardware import *
from delu.random import *
'''

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False

spelling_show_suggestions = True

# >>> HTML >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
html_static_path = ['_static']
html_favicon = 'images/favicon.ico'

# >>> Material Sphinx >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# This theme concatenates autosummary tables nicely and supports versioning.
import sphinx_material  # noqa

html_theme = 'sphinx_material'
html_css_files = ['material_theme.css']
html_logo = 'images/logo_28x28.svg'
html_theme_options = {
    # Full list of options: https://github.com/bashtage/sphinx-material/blob/master/sphinx_material/sphinx_material/theme.conf
    'base_url': 'https://yura52.github.io/delu',
    # Full list of colors (not all of them are available in sphinx-material, see theme.conf above):
    # https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/#primary-color
    # Nice colors: white, blue, red, deep purple (indigo in mkdocs)
    'color_primary': 'white',
    'globaltoc_depth': 1,
    # search here for logo icons: https://www.compart.com/en/unicode
    # 'logo_icon': '&#127968;',
    'nav_links': [],
    'nav_title': project + ' ' + version,
    'repo_name': project,
    'repo_url': repo_url,
    'repo_type': 'github',
    'master_doc': False,
    'version_dropdown': True,
    'version_json': '_static/versions.json',
}
html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html'],
}

# >>> Read the Docs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# extensions.append('sphinx_rtd_theme')
# html_theme = 'sphinx_rtd_theme'
# html_css_files = ['rtd_theme.css']
# html_logo = 'images/logo_120x120.svg'

# github_url = repo_url
# html_theme_options = {
#     'logo_only': True,
#     'display_version': False,
#     'prev_next_buttons_location': None,
#     'style_external_links': True,
# }

# >>> Furo >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NOTE
# - uncomment the "version" toctree in index.rst
# - rename "docs/version.rst.backup" to "docs/version.rst"
# extensions.append('sphinx_remove_toctrees')
# remove_from_toctrees = ['reference/api/*']

# html_title = project
# html_css_files = ['furo.css']
# html_logo = 'images/logo_120x120.svg'
# html_theme = 'furo'
# html_theme_options = {
#     'sidebar_hide_name': True,
#     'source_repository': 'https://github.com/Yura52/delu/',
#     'source_branch': 'main',
#     'source_directory': 'docs/',
#     'light_css_variables': {
#         'admonition-title-font-size': '90.5%',
#         'admonition-font-size': '90.5%',
#         'code-font-size': 'var(--font-size--small)',
#     },
# }
