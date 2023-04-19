#!/usr/bin/env python
import os
import sys
from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# Load the __version__ variable without importing the package already
exec(open("periodogram/version.py").read())

setup(
    name="periodogram",
    version=__version__,
    author="Daniel Hey",
    url="https://github.com/danhey/periodogram",
    packages=["periodogram"],
    description="Simple periodogram manipulation in Python.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=install_requires,
)
