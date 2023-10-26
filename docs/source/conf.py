import os
import os.path as osp
import sys
import datetime
import dbgym

import torch_geometric


sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('./reference'))
sys.setrecursionlimit(1500)

project = 'DBGym Team'
author = 'DBGym Team'
release = '0.0.1'
# version = dbgym.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

# extensions = [
#     'rinoh.frontend.sphinx',
#     'sphinx.ext.autodoc',
# ]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'

html_sidebars = {
    '**': [
        'globaltoc.html',  
        'sourcelink.html',  
        'searchbox.html',  
    ]
}


html_static_path = ['_static']

latex_elements = {
    'papersize':'letterpaper',
    'pointsize':'10pt',
    'preamble': '',
    'figure_align': "htbp"

}