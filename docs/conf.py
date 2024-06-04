import importlib
import shutil
from pathlib import Path

author = "dynamicslab"
project = "pysindy"  # package name


# no need to edit below this line

copyright = f"2020, {author}"

module = importlib.import_module(project)
version = release = getattr(module, "__version__")

master_doc = "index"

extensions = [
    "nbsphinx",
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

nb_execution_mode = "off"

apidoc_module_dir = f"../{project}"
apidoc_excluded_paths = ["tests"]
apidoc_toc_file = False

autodoc_default_options = {"members": True}
autodoc_member_order = "bysource"
autoclass_content = "init"

language = "en"

here = Path(__file__).parent.resolve()

if (here / "static/custom.css").exists():
    html_static_path = ["static"]

exclude_patterns = ["build", "_build", "Youtube"]
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

intersphinx_mapping = {
    "derivative": ("https://derivative.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------
# michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
from sphinx.ext.napoleon.docstring import GoogleDocstring  # noqa: E402


def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section


def patched_parse(self):
    """
    we now patch the parse method to guarantee that the the above methods are
    assigned to the _section dict
    """
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


def setup(app):
    """Our sphinx extension for copying from examples/ to docs/examples

    Since nbsphinx does not handle glob/regex paths, we need to
    manually copy documentation source files from examples.  See issue
    # 230.
    """
    doc_examples = here / "examples"
    if not doc_examples.exists():
        (here / "examples").mkdir()
    example_source = (here / "../examples").resolve()
    source_notebooks = example_source.glob("**/*.ipynb")
    shutil.copy(example_source / "README.rst", doc_examples / "index.rst")
    for notebook in source_notebooks:
        if notebook.parent == example_source:
            new_dir = doc_examples / notebook.stem
        else:
            new_dir = doc_examples / notebook.parent.stem
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / "example.ipynb"
        print(f"Creating file {new_file}")
        shutil.copy(notebook, new_file)
    # Notebook 15 uses an image file
    (doc_examples / "15_pysindy_lectures/data").mkdir(exist_ok=True)
    shutil.copy(
        example_source / "data/optimizer_summary.jpg",
        doc_examples / "15_pysindy_lectures/data/optimizer_summary.jpg",
    )
    if (here / "static/custom.css").exists():
        app.add_css_file("custom.css")
