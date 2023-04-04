from os.path import join
import re
import warnings

from setuptools_scm import Configuration
from setuptools_scm.utils import data_from_mime, trace
from setuptools_scm.version import meta, tags_to_versions


warnings.warn(DeprecationWarning(
    "This plugin is obsolete. setuptools_scm >= 7.0.0 supports Git archives by itself."
))


tag_re = re.compile(r'(?<=\btag: )([^,]+)\b')

# Define default config so call to meta() does not warn
config = Configuration()


def archival_to_version(data):
    trace('data', data)
    versions = tags_to_versions(tag_re.findall(data.get('ref-names', '')))
    if versions:
        return meta(versions[0], config=config)


def parse(root):
    archival = join(root, '.git_archival.txt')
    data = data_from_mime(archival)
    return archival_to_version(data)
