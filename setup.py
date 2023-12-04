import os
import re

from setuptools import setup, find_packages


def is_comment_or_empty(line):
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def remove_comments_and_empty_lines(lines):
    return [line for line in lines if not is_comment_or_empty(line)]


VERSION_RE = re.compile(r"""VERSION = ['"]([0-9.]+)['"]""")
ROOT = os.path.dirname(__file__)


def get_version():
    path = os.path.join(ROOT, "skorch_forecasting", "version.py")
    init = open(path).read()
    return VERSION_RE.search(init).group(1)


def get_requirements():
    with open("requirements.txt") as file:
        return remove_comments_and_empty_lines(file.read().splitlines())


def get_packages():
    return find_packages(exclude=["tests", "tests.*"])


python_requires = ">=3.10"

setup(
    name="skorch-forecasting",
    version=get_version(),
    install_requires=get_requirements(),
    packages=get_packages(),
    python_requires=python_requires,
)
