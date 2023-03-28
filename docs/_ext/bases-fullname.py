"""
Make inheritance unambiguous by showing full class name.

Add an attribute `__ambiguous_inheritance__: bool | list[type]` to your class.
"""

def process_bases(app, name, obj, options, bases):
    ambiguity = getattr(obj, "__ambiguous_inheritance__", False)
    if isinstance(ambiguity, bool):
        ambiguity = bases if ambiguity else ()
    for i, base in enumerate(bases):
        if base in ambiguity:
            bases[i] = ":class:`{}.{}`".format(base.__module__, base.__name__)


def setup(app):
    app.connect("autodoc-process-bases", process_bases)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
