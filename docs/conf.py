import datetime
import platform
import sys
from pathlib import Path

# Add the repository root to PYTHONPATH.
delu_path = Path.cwd()
while not (delu_path.name == 'delu' and delu_path.parent.name != 'delu'):
    delu_path = delu_path.parent
sys.path.append(str(delu_path))
import delu  # noqa

# >>> project >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
author = 'Yura52'
copyright = f'{datetime.datetime.now().year}, {author}'
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
    # libenchant is not available for Apple CPUs.
    extensions.append('sphinxcontrib.spelling')

# autodoc_typehints = 'description'
autodoc_typehints = 'signature'
autodoc_class_signature = 'separated'
maximum_signature_line_length = 88

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False

spelling_show_suggestions = True

# >>> HTML >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
html_favicon = 'images/favicon.ico'
html_logo = 'images/logo.svg'
html_static_path = ['_static']
html_title = project

# >>> Furo >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
html_css_files = ['custom.css']
html_theme = 'furo'
html_theme_options = {
    'sidebar_hide_name': True,
    'source_repository': 'https://github.com/Yura52/delu/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': repo_url,
            'html': '''
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            ''',  # noqa: E501
            'class': '',
        },
    ],
    'light_css_variables': {
        'admonition-title-font-size': '90.5%',
        'admonition-font-size': '90.5%',
        'code-font-size': 'var(--font-size--small)',
    },
}
