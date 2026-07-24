"""
Typst AST for generating CeTZ/Fletcher string diagrams.

Example output::

    #import "@preview/cetz:0.5.2": canvas, draw
    #import "@preview/fletcher:0.5.9" as fletcher

    #canvas({
      import draw: *
      bezier((0,0), (1,0), (0.3, -0.5), (0.7, 0.5))
      line((0.6, 0.5), (1.8, 0.5), (1.8, 1.1), (0.6, 1.1),
           close: true, fill: white, stroke: black + 0.5pt)
      content((1.2, 0.8), [$f$])
    })

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    TypstNode
    Str
    Int
    Float
    Coord
    Call
    ContentExpr
    Block
    Canvas
    Document
"""
from __future__ import annotations

from dataclasses import dataclass, field


class TypstNode:
    """Base class for Typst AST nodes."""
    def render(self, indent: int = 0) -> str:
        raise NotImplementedError


@dataclass
class Str(TypstNode):
    """A quoted string literal."""
    value: str

    def render(self, indent=0):
        return f'"{self.value}"'


@dataclass
class Int(TypstNode):
    """An integer literal."""
    value: int

    def render(self, indent=0):
        return str(self.value)


@dataclass
class Float(TypstNode):
    """A floating-point literal, rendered with 4 decimal places."""
    value: float
    prec: int = 4

    def render(self, indent=0):
        s = f"{round(self.value, self.prec):.{self.prec}f}"
        return s.rstrip('0').rstrip('.')


@dataclass
class Angle(TypstNode):
    """An angle literal, e.g. ``45deg``."""
    degrees: float

    def render(self, indent=0):
        return f"{Float(self.degrees).render()}deg"


@dataclass
class Pct(TypstNode):
    """A percentage literal, e.g. ``50%``."""
    value: float

    def render(self, indent=0):
        return f"{Float(self.value).render()}%"


@dataclass
class Bool(TypstNode):
    """A boolean literal: ``true`` or ``false``."""
    value: bool

    def render(self, indent=0):
        return "true" if self.value else "false"


@dataclass
class NoneVal(TypstNode):
    """The ``none`` literal."""

    def render(self, indent=0):
        return "none"


@dataclass
class Dict(TypstNode):
    """A Typst dictionary literal, e.g. ``(fill: white, stroke: black)``."""
    items: dict[str, TypstNode] = field(default_factory=dict)

    def render(self, indent=0):
        if not self.items:
            return "(:)"
        parts = [f"{k}: {v.render(indent)}" for k, v in self.items.items()]
        return f"({', '.join(parts)})"


@dataclass
class RawText(TypstNode):
    """Raw Typst code inserted verbatim into the output."""
    text: str

    def render(self, indent=0):
        pre = " " * (indent * 4)
        return pre + self.text


@dataclass
class Ident(TypstNode):
    """A Typst identifier, e.g. ``bezier``, ``draw``, ``canvas``."""
    name: str

    def render(self, indent=0):
        return self.name


@dataclass
class Attr(TypstNode):
    """An attribute access, e.g. ``draw.bezier``."""
    obj: TypstNode
    attr: str

    def render(self, indent=0):
        return f"{self.obj.render(indent)}.{self.attr}"


@dataclass
class Coord(TypstNode):
    """A 2D coordinate: ``(x, y)``."""
    x: float
    y: float
    prec: int = 4

    def render(self, indent=0):
        x = Float(self.x, self.prec).render()
        y = Float(self.y, self.prec).render()
        return f"({x}, {y})"


@dataclass
class Call(TypstNode):
    """A function call: ``func(arg1, arg2, kw=val)``."""
    func: TypstNode
    args: list[TypstNode] = field(default_factory=list)
    kwargs: dict[str, TypstNode] = field(default_factory=dict)

    def render(self, indent=0):
        pre = " " * (indent * 4)
        parts = [a.render(indent) for a in self.args]
        parts += [f"{k}: {v.render(indent)}" for k, v in self.kwargs.items()]
        inner = ", ".join(parts)
        return f"{pre}{self.func.render(indent)}({inner})"


@dataclass
class ContentExpr(TypstNode):
    """Typst content: ``[text]`` or ``[$math$]``."""
    text: str
    math: bool = False

    def render(self, indent=0):
        pre = " " * (indent * 4)
        inner = f"${self.text}$" if self.math else self.text
        return f"{pre}[{inner}]"


@dataclass
class LetBinding(TypstNode):
    """A let binding: ``let name = value``."""
    name: str
    value: TypstNode

    def render(self, indent=0):
        pre = " " * (indent * 4)
        return f"{pre}let {self.name} = {self.value.render(indent)}\n"


@dataclass
class Block(TypstNode):
    """A code block: ``{ stmt1; stmt2; ... }``."""
    body: list[TypstNode] = field(default_factory=list)
    semi: bool = False

    def render(self, indent=0):
        pre = " " * (indent * 4)
        if not self.body:
            return f"{pre}{{}}"
        lines = []
        for stmt in self.body:
            rendered = stmt.render(indent)
            if self.semi and not rendered.rstrip().endswith(";"):
                rendered = rendered.rstrip() + ";"
            if not rendered.endswith("\n"):
                rendered += "\n"
            lines.append(rendered)
        inner = "".join(lines)
        return f"{pre}{{\n{inner}{pre}}}"


@dataclass
class OnLayer(TypstNode):
    """An ``on-layer(N, { ... })`` call."""
    layer: int
    body: Block

    def render(self, indent=0):
        pre = " " * (indent * 4)
        body_str = self.body.render(indent + 1).rstrip()
        return f"{pre}on-layer({self.layer}, {body_str})\n"


@dataclass
class Import(TypstNode):
    """A ``#import`` statement."""
    path: str
    members: list[str] = field(default_factory=list)
    alias: str | None = None
    all_members: bool = False

    def render(self, indent=0):
        if self.all_members:
            return f'#import "{self.path}": *'
        if self.members:
            return f'#import "{self.path}": {", ".join(self.members)}'
        if self.alias:
            return f'#import "{self.path}" as {self.alias}'
        return f'#import "{self.path}"'


@dataclass
class Canvas(TypstNode):
    """A ``cetz.canvas({ ... })`` block."""
    canvas_kwargs: dict[str, TypstNode] = field(default_factory=dict)
    body: list[TypstNode] = field(default_factory=list)

    def render(self, indent=0):
        pre = " " * (indent * 4)
        kw_parts = [
            f"{k}: {v.render()}"
            for k, v in self.canvas_kwargs.items()]
        kw_str = (", ".join(kw_parts) + ", ") if kw_parts else ""
        body_lines = []
        for stmt in self.body:
            rendered = stmt.render(indent + 1)
            if not rendered.endswith("\n"):
                rendered += "\n"
            body_lines.append(rendered)
        inner = "".join(body_lines)
        return f"{pre}#canvas({kw_str}{{\n{inner}{pre}}})"


@dataclass
class Document(TypstNode):
    """Top-level Typst document with imports and a canvas."""
    imports: list[Import] = field(default_factory=list)
    preamble: list[TypstNode] = field(default_factory=list)
    content: TypstNode | None = None

    def render(self, indent=0):
        parts = [i.render(indent) + "\n" for i in self.imports]
        for node in self.preamble:
            parts.append(node.render(indent) + "\n")
        if self.content:
            parts.append("\n")
            parts.append(self.content.render(indent))
            parts.append("\n")
        return "".join(parts)
