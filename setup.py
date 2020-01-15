"""
Setup discopy package.
"""

import re
from setuptools import setup


def get_version():
    with open('__init__.py', 'r') as f:
        match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")


VERSION = get_version()

if __name__ == '__main__':  # pragma: no cover
    setup(name='discopy',
          version=VERSION,
          package_dir={'discopy': ''},
          packages=['discopy'],
          description='Distributional Compositional Python',
          long_description=open("README.md", "r").read(),
          long_description_content_type="text/markdown",
          url='https://github.com/oxford-quantum-group/discopy',
          author='Alexis Toumi',
          author_email='alexis.toumi@cs.ox.ac.uk',
          download_url='https://github.com/'
                       'oxford-quantum-group/discopy/archive/'
                       '{}.tar.gz'.format(VERSION),
          install_requires=['numpy',
                            'networkx',
                            'matplotlib',
                            'pytket'],
          )
