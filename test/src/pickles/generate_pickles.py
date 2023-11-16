"""
This file is meant as a convenient way to generate pickle files from the files
inside `/src`. These pickles are then used when tests are run to verify that
we can parse pickles of older version of DisCoPy.

Inside the `/src` folder, each file should include a variable named `pick`.
The value of this variable is then pickled into the associated pickle file.

`/main` contains pickles for the latest version
`/0.6` contains pickles for version 0.6

Note:
    To generate pickle files for later version, the given versions of DisCoPy
    needs to be installed and the `version` needs to be updated before run.
    This is easiest done if DisCoPy is installed with the `--editable` flag
    and the given version is checked out in git.
"""
import pickle
import pkgutil


version = "main"
pkgs = list(module for _, module, _ in pkgutil.iter_modules(["src"]))

for pkg in pkgs:
    impmodule = __import__(f"src.{pkg}")
    mod = list(impmodule.__dict__.values())[-1]
    fn = pkg.replace('_', '.')
    with open(f'./{version}/{fn}.pickle', 'wb') as f:
        print(mod.pick)
        pickle.dump(mod.pick, f)
