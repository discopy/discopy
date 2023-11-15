import pickle
import pkgutil

pkgs = list(module for _, module, _ in pkgutil.iter_modules(["src"]))

for pkg in pkgs:
    impmodule = __import__(f"src.{pkg}")
    mod = list(impmodule.__dict__.values())[-1]
    fn = pkg.replace('_', '.')
    with open(f'./main/{fn}.pickle', 'wb') as f:
        print(mod.pick)
        pickle.dump(mod.pick, f)
