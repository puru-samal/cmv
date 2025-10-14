#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='cmv',
    version='0.0.0',
    description='A voice conversion project',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='',
    install_requires=['torchaudio'],
    packages=find_packages(),
    python_requires='>=3.12',
)

