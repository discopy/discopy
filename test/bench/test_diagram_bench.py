"""Integration benchmarks for DisCoPy.

Run:
    uv run pytest test/bench/ --codspeed -v
"""
from discopy.monoidal import Box, Functor, Id, Ty

_X = Ty('x')
_BOXES = [Box(f'f{i}', _X, _X) for i in range(100)]
_DIAGRAM = Id(_X).then(*_BOXES)

_Y = Ty('y')
_BOXES_Y = [Box(f'g{i}', _Y, _Y) for i in range(100)]
_FUNCTOR = Functor(ob={_X: _Y}, ar=dict(zip(_BOXES, _BOXES_Y)))


def test_chain_compose(benchmark):
    """Sequential composition of 100 boxes."""
    benchmark(lambda: Id(_X).then(*_BOXES))


def test_functor_eval(benchmark):
    """Apply a monoidal Functor to a 100-box chain diagram."""
    benchmark(lambda: _FUNCTOR(_DIAGRAM))
