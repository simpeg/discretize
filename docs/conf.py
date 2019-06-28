# -*- coding: utf-8 -*-
#
# discretize documentation build configuration file, created by
# sphinx-quickstart on Fri Aug 30 18:42:44 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
from datetime import datetime
import discretize
import subprocess
from sphinx_gallery.sorting import FileNameSortKey

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))
# sys.path.append(os.path.pardir)


# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = False

numpydoc_class_members_toctree = False

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'discretize'
copyright = u'2013 - {}, SimPEG Developers, http://simpeg.xyz'.format(
    datetime.now().year
)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.4.5'
# The full version, including alpha/beta/rc tags.
release = '0.4.5'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

linkcheck_ignore = [
    'http://math.lanl.gov/~mac/papers/numerics/HS99B.pdf',
    'http://wiki.python.org/moin/NumericAndScientific',
    'http://wiki.python.org/moin/PythonEditors',
    'http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html#numpy.array',
    'http://dx.doi.org/10.1016/j.cageo.2015.09.015',
    'http://slack.simpeg.xyz',
]

linkcheck_retries = 3
linkcheck_timeout = 500

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# API doc options
apidoc_module_dir = '../discretize'
apidoc_output_dir = 'api/generated'
apidoc_toc_file = False
apidoc_excluded_paths = []
apidoc_separate_modules = True
# apidoc_extra_args = ['-t _templates']


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    pass
except Exception:
    html_theme = 'default'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = './images/logo-block.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'discretize'


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'discretize.tex', u'discretize documentation',
   u'SimPEG Developers', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'simpeg', u'discretize Documentation',
     [u'SimPEG Developers'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'properties': ('https://propertiespy.readthedocs.io/en/latest/', None),
    'pyvista': ('http://docs.pyvista.org/', None),
}


# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'discretize', u'discretize documentation',
    u'SimPEG Developers', 'discretize', 'Finite volume methods for python.',
    'Miscellaneous'),
]


# Use pyvista's image scraper for example gallery
import pyvista
# Make sure off screen is set to true when building locally
pyvista.OFF_SCREEN = True

# Sphinx Gallery
sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': ['../examples',
                      '../tutorials/mesh_generation',
                      '../tutorials/operators',
                      '../tutorials/inner_products',
                      '../tutorials/pde'
                      ],
    'gallery_dirs': ['examples',
                     'tutorials/mesh_generation',
                     'tutorials/operators',
                     'tutorials/inner_products',
                     'tutorials/pde'
                     ],
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '\.py',
    'backreferences_dir': 'api/generated/backreferences',
    'doc_module': 'discretize',
    # 'reference_url': {'discretize': None},
}
# Do not run or scrape `pyvista` examples on Python 2 because sphinx gallery
# doesn't support custom scrapers. But also, data was pickled in Python 3
if sys.version_info[0] >= 3:
    # Requires pyvista>=0.18.0 and Python 3
    sphinx_gallery_conf["image_scrapers"] = (pyvista.Scraper(), 'matplotlib')
else:
    # Don't run pyvista examples at all on Python 2
    # Primarily because data used for it was pickled in Python 3
    sphinx_gallery_conf["filename_pattern"] = r"plot_(?!pyvista)\.py",

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

html_context = {
    'menu_links_name': 'Getting connected',
    'menu_links': [
        ('<i class="fa fa-external-link-square fa-fw"></i> SimPEG', 'https://simpeg.xyz'),
        # ('<i class="fa fa-gavel fa-fw"></i> Code of Conduct', 'https://github.com/fatiando/verde/blob/master/CODE_OF_CONDUCT.md'),
        ('<i class="fa fa-comment fa-fw"></i> Contact', 'http://slack.simpeg.xyz'),
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/simpeg/discretize'),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'doc',
    'galleries': sphinx_gallery_conf['gallery_dirs'],
    'gallery_dir': dict(zip(sphinx_gallery_conf['gallery_dirs'],
                            sphinx_gallery_conf['examples_dirs'])),
    'github_repo': 'simpeg/discretize',
    'github_version': 'master',
}

autodoc_member_order = 'bysource'

nitpick_ignore = [
    ('py:class', 'discretize.CurvilinearMesh.Array'),
    ('py:class', 'discretize.mixins.vtkModule.InterfaceTensorread_vtk'),
    ('py:class', 'callable'),
    ('py:obj', 'vtk.vtkDataSet'),
]


## Build the API
dirname, filename = os.path.split(os.path.abspath(__file__))
subprocess.run([
    "sphinx-autogen", "-i", "-t",
    os.path.sep.join([dirname, "_templates"]),
    "-o", os.path.sep.join([dirname,"api/generated"]),
    os.path.sep.join([dirname,"api/index.rst"])
])
