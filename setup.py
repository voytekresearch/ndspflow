"""ndspflow setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('ndspflow', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'ndspflow',
    version = '0.0.1',
    description = 'neural dsp nipype workflow',
    long_description = long_description,
    python_requires = '>=3.5',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    maintainer = 'Ryan Hammonds',
    maintainer_email = 'rhammonds@ucsd.edu',
    url = 'https://github.com/voytekresearch/ndspflow',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    platforms = 'any',
    project_urls = {
        'Documentation' : 'https://neurodsp-tools.github.io/',
        'Source' : 'https://github.com/voytekresearch/ndspflow',
        'Bug Reports' : 'https://github.com/voytekresearch/ndspflow/issues',
    },
    download_url = 'https://github.com/voytekresearch/ndspflow/releases',
    keywords = ['neuroscience', 'neural oscillations', 'power spectra', '1/f', 'electrophysiology'],
    install_requires = install_requires,
    tests_require = ['pytest'],
    extras_require = {
        'tests'   : ['pytest']
    }
)
