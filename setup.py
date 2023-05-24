"""
Setup discopy package.
"""

if __name__ == '__main__':  # pragma: no cover
    import pathlib
    from re import search, M
    from setuptools import setup, find_packages

    def get_version(filename="discopy/__init__.py",
                    pattern=r"^__version__ = ['\"]([^'\"]*)['\"]"):
        with open(filename, 'r') as file:
            MATCH = search(pattern, file.read(), M)
            if MATCH:
                return MATCH.group(1)
            else:
                raise RuntimeError("Unable to find version string.")

    VERSION = get_version()

    def get_reqs(filename):
        try:
            with pathlib.Path(filename).open() as file:
                return [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            from warnings import warn
            warn("{} not found".format(filename))
            return []

    REQS = get_reqs("requirements.txt")
    TEST_REQS = get_reqs("test/requirements.txt")
    DOCS_REQS = get_reqs("docs/requirements.txt")

    README = open("README.md", "r").read()

    setup(name='discopy',
          version=VERSION,
          package_dir={'discopy': 'discopy'},
          packages=find_packages(),
          description='The Python toolkit for computing with string diagrams.',
          long_description=README,
          long_description_content_type="text/markdown",
          url='https://discopy.org',
          project_urls={
            'Documentation': 'https://docs.discopy.org',
            'Source': 'https://github.com/discopy/discopy',
            'Tracker': 'https://github.com/discopy/discopy/issues',
          },
          keywords='diagrams category-theory quantum-computing nlp',
          author='Alexis Toumi',
          author_email='alexis@toumi.email',
          download_url='https://github.com/discopy/discopy/archive/'
                       f'{VERSION}.tar.gz',
          install_requires=REQS,
          tests_require=TEST_REQS,
          extras_require={'test': TEST_REQS, 'docs': DOCS_REQS},
          data_file=[('test', ['test/requirements.txt']),
                     ('docs', ['docs/requirements.txt'])],
          python_requires='>=3.9',
          )
