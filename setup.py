#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: setup.py.py
"""

import setuptools
from os import path
import re
from codecs import open

packages_name = 'TART'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, packages_name, '__init__.py'), encoding='utf-8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=packages_name,
    version=version,
    author="Shiyu Huang",
    author_email="huangsy13@gmail.com",
    description=packages_name+" program",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ],

    keywords=packages_name,

    packages=setuptools.find_packages(exclude=['examples']),

    install_requires=['numpy',
                      'six'],
)   
