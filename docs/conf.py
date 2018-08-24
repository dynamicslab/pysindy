import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../"))
import sparsereg

project = "Sparsereg"
copyright = "{}, Markus Quade".format(datetime.datetime.now().year)
author = "Markus Quade"
version = release = sparsereg.__version__

master_doc = "index"

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

apidoc_module_dir = "../sparsereg"
apidoc_excluded_paths = ["tests"]

autodoc_default_flags = ["members"]
autodoc_member_order = "bysource"
autoclass_content = "init"

language = None

exclude_patterns = ["_build"]
pygments_style = "sphinx"

add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = "any"

# templates_path = ["templates"]
# html_sidebars = {
#     "index": ["about.html", "sidebarintro.html", "navigation.html", "searchbox.html", "hacks.html"],
#     "**": ["about.html", "navigation.html", "localtoc.html", "searchbox.html", "hacks.html"],
# }
