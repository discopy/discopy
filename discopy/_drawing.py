# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from discopy.monoidal import Ty, Box


@dataclass
class Cell:
    start: int
    end: int
    label: Ty | Box = None

    def __add__(self, offset: int) -> Cell:
        return Cell(self.start + offset, self.end + offset, self.label)

    def __sub__(self, offset: int) -> Cell:
        return self + (-offset)

    def __str__(self):
        return f"Cell({self.start}, {self.end}, {self.label})"


class Wire(Cell):
    def __init__(self, start: int, label: Ty = None):
        super().__init__(start, start, label)

    def __add__(self, offset: int) -> Wire:
        return Wire(self.start + offset, self.label)

    def __str__(self):
        return f"Wire({self.start}, {self.label})"


@dataclass
class Embedding:
    """
    >>> from discopy.monoidal import *
    >>> x = Ty('x')
    >>> cap = Box('-', Ty(), x @ x)
    >>> cup = Box('-', x @ x, Ty())
    >>> diagram = Diagram((Layer(Ty(), cap, Ty(), cap, Ty()), ), Ty(), x ** 4)
    >>> print(Embedding.from_diagram(diagram))
    """
    rows: list[list[Cell]]

    def __str__(self):
        result = ""
        self = self - self.min
        width = self.max
        for row in self:
            row_str = ""
            for cell in row:
                row_str += (cell.start - len(row_str)) * " "
                if isinstance(cell, Wire):
                    row_str += "|"
                else:
                    row_str += " " + (
                        cell.end - cell.start - 1) * str(cell.label)[0] + " "
            result += row_str + (width - len(row_str) + 1) * " " + "\n"
        return result

    @property
    def min(self):
        return min([min([0] + [cell.start for cell in row]) for row in self.rows])

    @property
    def max(self):
        return max([max([0] + [cell.end for cell in row]) for row in self.rows])

    def __add__(self, offset: int):
        return Embedding([
            [cell + offset for cell in row] for row in self.rows])

    __sub__ = Cell.__sub__

    def __iter__(self):
        for row in self.rows:
            yield row

    @staticmethod
    def from_diagram(diagram):
        def make_space(rows, limit, space):
            if space < 0:
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        cell.start += space * int(cell.start <= limit)
                        cell.end += space * int(cell.end <= limit)
            if space > 0:
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        cell.start += space * int(cell.start >= limit)
                        cell.end += space * int(cell.end >= limit)
        rows = [[Wire(i + 1, x) for i, x in enumerate(diagram.dom)]]
        for layer in diagram.inside:
            offset = 0
            rows.append([])
            rows.append([Wire(cell.start, cell.label) for cell in rows[-2]])
            boxes, wires = rows[-2], rows[-1]
            for i, type_or_box in enumerate(layer):
                if not i % 2:
                    for j, x in enumerate(type_or_box):
                        boxes.append(Wire(wires[offset + j].start, x))
                    offset += len(type_or_box)
                else:
                    box = type_or_box
                    if not box.dom:
                        if not wires:
                            start = 0
                        elif not offset:
                            start = wires[0].start - len(box.cod) - 3
                        else:
                            start = wires[offset - 1].end
                        end = start + len(box.cod) + 1
                    else:
                        start = wires[offset].start
                        end = max(wires[offset + len(box.dom) - 1].end + 1,
                                  start + len(box.cod) + 1)
                    if offset:
                        left = boxes[-1].end
                        make_space(rows, left, min(0, start - left - 1))
                    if offset + len(box.dom) < len(wires):
                        right = wires[offset + len(box.dom)].start - 1
                        make_space(rows, right, max(0, end - right + 1))
                    boxes.append(Cell(start, end, box))
                    rows[-1] = wires = wires[:offset]\
                        + [Wire(start + j + 1, x) for j, x in enumerate(box.cod)]\
                        + wires[offset + len(box.dom):]
                    offset += len(box.cod)
        return Embedding(rows)
