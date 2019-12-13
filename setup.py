"""
Setup discopy package.
"""

from setuptools import setup
from config import VERSION


if __name__ == '__main__':
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
                       '0.1.2b.tar.gz',
          install_requires=['numpy', 'pytket', 'jax', 'jaxlib'],
          )
