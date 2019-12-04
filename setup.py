from setuptools import setup
from config import VERSION

setup(name='discopy',
      version=VERSION,
      package_dir={'discopy': ''},
      packages=['discopy'],
      description='Distributional Compositional Python',
      url='https://github.com/oxford-quantum-group/discopy',
      author='Alexis Toumi',
      author_email='alexis.toumi@cs.ox.ac.uk',
      download_url='https://github.com/oxford-quantum-group/discopy/archive/'
                   '0.1.1.tar.gz',
      install_requires=['numpy', 'pytket', 'jax', 'jaxlib'],
      )
