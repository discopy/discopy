# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from lxml.html import Element
from lxml.etree import SubElement


@dataclass
class Cell:
    """
    A cell is a pair of integers ``start`` and ``stop``
    and an optional ``label``.

    Parameters:
        start : The left of the cell.
        stop : The right of the cell.
        label : The label of the cell.
    """
    start: int
    stop: int
    label: Ty | Box = None

    def __add__(self, offset: int) -> Cell:
        return Cell(self.start + offset, self.stop + offset, self.label)

    def __sub__(self, offset: int) -> Cell:
        return self + (-offset)

    def __str__(self):
        return f"Cell({self.start}, {self.stop}, {self.label})"


class Wire(Cell):
    """ A wire is a cell with ``stop = start``. """
    def __init__(self, start: int, label: Ty = None):
        super().__init__(start, start, label)

    def __add__(self, offset: int) -> Wire:
        return Wire(self.start + offset, self.label)

    def __str__(self):
        return f"Wire({self.start}, {self.label})"


@dataclass
class Grid:
    """
    A grid is a list of rows, a row is a list of cells.

    Parameters:
        rows : The list of lists of cells inside the grid.
    """
    rows: list[list[Cell]]

    def to_html(self) -> Element:
        """
        Turn a grid into an html table.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> x = Ty('x')
        >>> f = Box('f', x, x @ x)
        >>> diagram = (f @ f[::-1]).foliation()
        >>> grid = Grid.from_diagram(diagram)

        >>> from lxml.etree import tostring
        >>> pprint = lambda x: print(
        ...     tostring(x, pretty_print=True).decode('utf-8').strip())
        >>> pprint(grid.to_html())
        <table>
          <tr>
            <td colspan="1" class="wire">x</td>
            <td colspan="7" class="wire">x</td>
            <td colspan="9" class="wire">x</td>
          </tr>
          <tr>
            <td colspan="4" class="box">f</td>
            <td colspan="6">f</td>
            <td colspan="4" class="box">f</td>
          </tr>
          <tr>
            <td colspan="1" class="wire">x</td>
            <td colspan="3" class="wire">x</td>
            <td colspan="7" class="wire">x</td>
          </tr>
        </table>
        """
        table = Element("table")
        for row in self.rows:
            tr = SubElement(table, "tr")
            offset = 0
            for cell in row:
                if cell.start > offset:
                    td = SubElement(tr, "td")
                    td.text = cell.label.name
                    td.set('colspan', str(cell.start - offset))
                    if cell.start == cell.stop:
                        td.set("class", "wire")
                if cell.start < cell.stop:
                    td = SubElement(tr, "td")
                    td.text = cell.label.name
                    td.set("colspan", str(cell.stop - cell.start))
                    td.set("class", "box")
        return table

    def to_ascii(self) -> str:
        """
        Turn a grid into an ascii drawing.

        Examples
        --------
        >>> from discopy.monoidal import *
        >>> x = Ty('x')
        >>> f = Box('f', x, x @ x)
        >>> diagram = (f @ f[::-1] >> f @ f[::-1]).foliation()
        >>> print(Grid.from_diagram(diagram).to_ascii())
         |         | |
         ---f---   -f-
         |     |   |
         -f-   --f--
         | |   |
        >>> cup, cap = Box('-', x @ x, Ty()), Box('-', Ty(), x @ x)
        >>> unit = Box('o', Ty(), x)
        >>> spiral = cap >> cap @ x @ x >> x @ x @ x @ unit @ x\\
        ...     >> x @ cap @ x @ x @ x @ x >> x @ x @ unit[::-1] @ x @ x @ x @ x\\
        ...     >> x @ cup @ x @ x @ x >> x @ cup @ x >> cup
        >>> print(Grid.from_diagram(spiral).to_ascii())
                     -------
                     |     |
         ----------  |     |
         |        |  |     |
         |        |  |  o  |
         |        |  |  |  |
         |  ----  |  |  |  |
         |  |  |  |  |  |  |
         |  |  o  |  |  |  |
         |  |     |  |  |  |
         |  -------  |  |  |
         |           |  |  |
         |           ----  |
         |                 |
         -------------------
        """
        def row_to_ascii(row):
            """ Turn a row into an ascii drawing. """
            result = ""
            for cell in row:
                result += (cell.start - len(result)) * " "
                if isinstance(cell, Wire):
                    result += "|"
                else:
                    result += " " + str(cell.label.name).center(
                        cell.stop - cell.start - 1, "-")
            return result
        return '\n'.join(map(row_to_ascii, self.rows)).strip('\n')

    def __add__(self, offset: int):
        return Grid([
            [cell + offset for cell in row] for row in self.rows])

    __sub__ = Cell.__sub__

    @staticmethod
    def from_diagram(diagram: Diagram) -> Grid:
        """
        Layout a diagram on a grid.

        The first row is a list of :class:`Wire` cells, then for each layer of
        the diagram there are two rows: the first for the boxes and the wires
        in between them, the second is a list of :class:`Wire` for the outputs.

        Parameters:
            diagram : The diagram to layout on a grid.
        """
        def make_space(rows: list[list[Cell]], limit: int, space: int) -> None:
            if space < 0:
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        cell.start += space * int(cell.start <= limit)
                        cell.stop += space * int(cell.stop <= limit)
            if space > 0:
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        cell.start += space * int(cell.start >= limit)
                        cell.stop += space * int(cell.stop >= limit)
        rows = [[Wire(2 * i + 1, x) for i, x in enumerate(diagram.dom)]]
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
                    width = 2 * (len(box.cod) or 1)
                    if not box.dom:
                        if not wires:
                            start = 0
                        elif not offset:
                            start = wires[0].start - width
                        else:
                            start = wires[offset - 1].stop + 2
                        stop = start + width
                    else:
                        start = wires[offset].start - 1
                        stop = wires[offset + len(box.dom) - 1].stop + 1
                        stop = max(stop, start + width)
                    if offset:
                        left = boxes[-1].stop
                        make_space(rows, left, min(0, start - left - 2))
                    if offset + len(box.dom) < len(wires):
                        right = wires[offset + len(box.dom)].start - 1
                        make_space(rows, right, max(0, stop - right + 1))
                    boxes.append(Cell(start, stop, box))
                    rows[-1] = wires = wires[:offset]\
                        + [Wire(start + 2 * j + 1, x)
                           for j, x in enumerate(box.cod)]\
                        + wires[offset + len(box.dom):]
                    offset += len(box.cod)
        offset = min([min([0] + [cell.start for cell in row]) for row in rows])
        return Grid(rows) - offset
