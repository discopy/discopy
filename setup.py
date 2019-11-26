from setuptools import setup
from config import VERSION

setup(name='discopy',
      version=VERSION,
      package_dir={'discopy': ''},
      packages=['discopy'],
      description='Distributional Compositional Python',
      url='https://github.com/toumix/discopy',
      author='Alexis Toumi',
      author_email='alexis.toumi@gmail.com',
      download_url='https://github.com/toumix/discopy/archive/0.0.1.3.tar.gz',
      install_requires=['numpy'],
      )
