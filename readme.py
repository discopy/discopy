from discopy.monoidal import Ty, Box
X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
f, g = Box("f", X @ Y, Z), Box("g", X, Y @ Z)
diagram = X @ g >> f @ Z

from discopy import monoidal, python
F = monoidal.Functor(
  {X: str, Y: list[str], Z: []},
  {f: lambda x, xs: print(", ".join([x] + xs)), g: lambda x: x.split()},
  cod=python.Function)
F(diagram)("Hello", "world!")  # Prints "Hello, world!" returns `(None, None)`