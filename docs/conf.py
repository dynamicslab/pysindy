import datetime
import importlib
import pathlib

author = "dynamicslab"
project = "pysindy"  # package name


# no need to edit below this line

copyright = f"{datetime.datetime.now().year}, {author}"

module = importlib.import_module(project)
version = release = getattr(module, "__version__")

master_doc = "index"

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_nbexamples",
]

apidoc_module_dir = f"../{project}"
apidoc_excluded_paths = ["tests"]
apidoc_toc_file = False

autodoc_default_options = {"members": True}
autodoc_member_order = "bysource"
autoclass_content = "init"

language = None

here = pathlib.Path(__file__).parent

if (here / "static/custom.css").exists():

    html_static_path = ["static"]

    def setup(app):
        app.add_stylesheet("custom.css")


exclude_patterns = ["build", "_build"]
# pygments_style = "sphinx"

add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = "any"
html_sourcelink_suffix = ""

example_gallery_config = dict(
    dont_preprocess=True,
    examples_dirs=["../examples"],
    gallery_dirs=["examples"],
    pattern=".+.ipynb",
)
