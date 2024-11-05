import importlib
import os
import re
import shutil
from pathlib import Path
from typing import TypeVar

import requests
import yaml
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.directives.other import TocTree
from sphinx.util.docutils import SphinxDirective

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

html_static_path = ["_static"]

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


def setup(app: Sphinx):
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
        print(f"Copying {notebook} to {new_file}")
        shutil.copy(notebook, new_file)
    # Notebook 15 uses an image file
    (doc_examples / "15_pysindy_lectures/data").mkdir(exist_ok=True)
    shutil.copy(
        example_source / "data/optimizer_summary.jpg",
        doc_examples / "15_pysindy_lectures/data/optimizer_summary.jpg",
    )
    if (here / "static/custom.css").exists():
        app.add_css_file("custom.css")

    _grab_external_examples(example_source, doc_examples)
    app.add_directive("pysindy-example", PysindyExample)


EXTERNAL_EXAMPLES: dict[str, dict[str, os.PathLike]] = {}


def _grab_external_examples(example_source: Path, doc_examples: Path):
    ext_config = example_source / "external.yml"
    with open(ext_config) as f:
        ext_examples = yaml.safe_load(f)
    for example in ext_examples:
        repo = example["repo"]
        ref = example["ref"]
        base = f"https://raw.githubusercontent.com/{repo}/{ref}/docs/build/"
        documents = fetch_notebook_list(base)
        file_map = {}
        for name, url in documents:
            file_map[name] = copy_html(base, url, repo)
        EXTERNAL_EXAMPLES[example["key"]] = file_map


class PysindyExample(SphinxDirective):
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"key": str, "title": str}
    has_content = True

    def run(self) -> list[nodes.Node]:
        key = self.options["key"]
        example_node = nodes.subtitle(text=self.options.get("title"))
        content_node = nodes.paragraph(text="\n".join(self.content))
        toc_items = [
            f"{name} <{relpath}>" for name, relpath in EXTERNAL_EXAMPLES[key].items()
        ]
        toc_nodes = TocTree(
            name="PysindyExample",
            options={},
            arguments=[],
            content=StringList(initlist=toc_items),
            lineno=self.lineno,
            block_text="",
            content_offset=self.content_offset,
            state=self.state,
            state_machine=self.state_machine,
        ).run()
        return [example_node, content_node, *toc_nodes]


def fetch_notebook_list(base: str) -> list[tuple[str, str]]:
    """Gets the list of example notebooks from a repo's index.html

    Each entry is a tuple of the title name of a link and the address
    """

    index = requests.get(base + "index.html")
    if index.status_code != 200:
        raise RuntimeError("Unable to locate external example directory")
    text = str(index.content, encoding="utf-8")
    start = '<li class="toctree-l1"><a class="reference internal" href="'
    mid = '">'
    end = "</a></li>\n"
    matchstr = start + "(.*)" + mid + "(.*)" + end
    T = TypeVar("T")

    def deduplicate(mylist: list[T]) -> list[T]:
        return list(set(mylist))

    rellinks: list[str] = deduplicate(re.findall(matchstr, text))
    return [(name, address) for address, name in rellinks]


def copy_html(base: str, location: str, repo: str) -> str:
    """Create a local copy of external file, returning relative reference"""
    example_dir = Path(__file__).parent / "examples"
    repo_root = Path(__file__).parent / "_static" / repo
    repo_root.mkdir(parents=True, exist_ok=True)
    page = requests.get(base + location)
    if page.status_code != 200:
        raise RuntimeError("Unable to locate external example notebook")
    filename = repo_root / location.rsplit("/", 1)[1]
    with open(filename, "wb") as f:
        f.write(page.content)
    return os.path.relpath(filename, start=example_dir)
