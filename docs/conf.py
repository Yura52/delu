import datetime
import sys
from pathlib import Path

# Add the repository root to PYTHONPATH
zero_path = Path.cwd()
while not (zero_path.name == 'zero' and zero_path.parent.name != 'zero'):
    zero_path = zero_path.parent
sys.path.append(str(zero_path))
import zero  # noqa

# >>> Project information <<<
author = 'Yura52'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)
project = 'Zero'
release = zero.__version__
version = zero.__version__

# >>> General options <<<
default_role = 'py:obj'
pygments_style = 'default'
repo_url = 'https://github.com/Yura52/zero'
templates_path = ['_templates']

# >>> Extensions options <<<
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.spelling',
    'sphinx_copybutton',
    # 'sphinx_rtd_theme',
]

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

autosummary_generate = True

doctest_global_setup = '''
import numpy as np
import torch
import zero
from zero import *
from zero.data import *
from zero.hardware import *
from zero.random import *
'''

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False

spelling_show_suggestions = True

# >>> HTML and theme options <<<
html_static_path = ['_static']
html_favicon = 'images/favicon.ico'

# This theme concatenates autosummary tables nicely and supports versioning.
import sphinx_material  # noqa

html_theme = 'sphinx_material'
html_css_files = ['material_theme.css']
html_logo = 'images/logo_28x28.svg'
html_theme_options = {
    # Full list of options: https://github.com/bashtage/sphinx-material/blob/master/sphinx_material/sphinx_material/theme.conf
    'base_url': 'https://yura52.github.io/zero',
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
