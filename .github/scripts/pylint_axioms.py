"""
A pylint plugin reading a law stated with ``@axiom`` as the classmethod it
is, see ``discopy.axioms``.

The decorator returns an ``Axiom``, a descriptor binding the law to the
class it is accessed on, so the first parameter of a law is its category.
Astroid infers a classmethod from the ``classmethod`` decorator alone, so
without this transform pylint reads every law as a method wanting ``self``
and the members of its category as those of an instance.
"""

from astroid import MANAGER, nodes


def is_axiom(function: nodes.FunctionDef) -> bool:
    """ Whether the function is decorated with ``axiom``. """
    return function.decorators is not None and any(
        isinstance(decorator, nodes.Name) and decorator.name == "axiom"
        for decorator in function.decorators.nodes)


def as_classmethod(function: nodes.FunctionDef) -> None:
    """ Have astroid infer the function as a classmethod. """
    function.type = "classmethod"


def register(linter) -> None:
    """ The entry point of a plugin: the transform is registered on import. """


MANAGER.register_transform(nodes.FunctionDef, as_classmethod, is_axiom)
