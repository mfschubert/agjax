"""Setup `agjax`."""
import codecs
import os.path
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="agjax",
    version=get_version("agjax/__init__.py"),
    license="MIT",
    author="Martin F. Schubert",
    author_email="mfschubert@gmail.com",
    install_requires=[
        "autograd",
        "jax",
        "numpy",
        "parameterized",
        "pytest",
        "pytest-xdist",
    ],
    url="https://github.com/mfschubert/agjax",
    packages=setuptools.find_packages(),
    python_requires=">=3",
)