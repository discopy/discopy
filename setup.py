"""
Setup discopy package.
"""

if __name__ == '__main__':  # pragma: no cover
    from re import search, M
    from setuptools import setup, find_packages

    with open('discopy/__init__.py', 'r') as file:
        MATCH = search(r"^__version__ = ['\"]([^'\"]*)['\"]", file.read(), M)
        if MATCH:
            VERSION = MATCH.group(1)
        else:
            raise RuntimeError("Unable to find version string.")

    try:
        with open('test/requirements.txt', 'r') as file:
            TEST_REQ = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        from warnings import warn
        warn("test/requirements.txt not found")
        TEST_REQ = []

    setup(name='discopy',
          version=VERSION,
          package_dir={'discopy': 'discopy'},
          packages=find_packages(),
          description='Distributional Compositional Python',
          long_description=open("README.md", "r").read(),
          long_description_content_type="text/markdown",
          url='https://github.com/oxford-quantum-group/discopy',
          author='Alexis Toumi',
          author_email='alexis.toumi@cs.ox.ac.uk',
          download_url='https://github.com/'
                       'oxford-quantum-group/discopy/archive/'
                       '{}.tar.gz'.format(VERSION),
          install_requires=[
              l.strip() for l in open('requirements.txt').readlines()],
          tests_require=TEST_REQ,
          extras_require={'test': TEST_REQ},
          data_file=[('test', ['test/requirements.txt'])],
          python_requires='>=3',
          )
