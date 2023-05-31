# -*- coding: utf-8 -*-
"""
DisCoPy's grid drawing.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Cell
    Wire
    Grid
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discopy import monoidal
    import lxml

TABLE_STYLE = ".diagram .wire { border-left: 4px solid; text-align: left; } "\
              ".diagram .box { border: 4px solid; text-align: center; }"\
              ".diagram td { min-width: 20px; height: 20px; }"


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
    label: monoidal.Ty | monoidal.Box = None

    def __add__(self, offset: int) -> Cell:
        return Cell(self.start + offset, self.stop + offset, self.label)

    def __sub__(self, offset: int) -> Cell:
        return self + (-offset)

    def __str__(self):
        return f"Cell({self.start}, {self.stop}, {self.label})"


class Wire(Cell):
    """ A wire is a cell with ``stop = start``. """

    def __init__(self, start: int, label: monoidal.Ty = None):
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

    >>> from discopy.monoidal import *
    >>> x = Ty('x')
    >>> cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    >>> snake = x @ cap >> cup @ x
    >>> grid = Grid.from_diagram(snake)
    >>> print(grid)
    Grid([Wire(1, x)],
         [Wire(1, x), Cell(3, 8, cap)],
         [Wire(1, x), Wire(4, x), Wire(7, x)],
         [Cell(0, 5, cup), Wire(7, x)],
         [Wire(7, x)])
    """
    rows: list[list[Cell]]

    @property
    def max(self) -> int:
        """ The maximum horizontal coordinate of a grid. """
        return max(
            [max([0] + [cell.stop for cell in row]) for row in self.rows])

    @property
    def min(self) -> int:
        """ The minimum horizontal coordinate of a grid. """
        return min(
            [min([0] + [cell.start for cell in row]) for row in self.rows])

    def to_html(self) -> lxml.etree.ElementTree:
        """
        Turn a grid into an html table.

        Notes
        -----
        This function requires the `lxml` package to be installed in addition
        to the default requirements.

        Examples
        --------
        >>> from discopy.monoidal import *
        >>> x = Ty('x')
        >>> cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
        >>> unit = Box('unit', Ty(), x)
        >>> snake = x @ cap >> cup @ x
        >>> table = snake.to_grid().to_html()

        >>> from lxml.etree import tostring
        >>> print(tostring(table, pretty_print=True
        ...     ).decode().strip())  # doctest: +ELLIPSIS
        <div>
          <style>.diagram .wire { border-left: 4px solid; ...</style>
          <table class="diagram">
            <tr>
              <td class="wire">x</td>
              <td/>
              <td/>
              <td/>
              <td/>
              <td/>
              <td/>
            </tr>
            <tr>
              <td colspan="1"/>
              <td class="wire" colspan="2"/>
              <td class="box" colspan="5">cap</td>
            </tr>
            <tr>
              <td colspan="1"/>
              <td class="wire" colspan="3"/>
              <td class="wire" colspan="3">x</td>
              <td class="wire" colspan="1">x</td>
            </tr>
            <tr>
              <td class="box" colspan="5">cup</td>
              <td colspan="2"/>
              <td class="wire" colspan="1"/>
            </tr>
            <tr>
              <td colspan="7"/>
              <td class="wire" colspan="1"/>
            </tr>
          </table>
        </div>

        >>> spiral = cap >> cap @ x @ x >> x @ x @ x @ unit @ x\\
        ...     >> x @ cap @ x @ x @ x @ x\\
        ...     >> x @ x @ unit[::-1] @ x @ x @ x @ x\\
        ...     >> x @ cup @ x @ x @ x >> x @ cup @ x >> cup
        >>> spiral.to_grid().to_html().write(
        ...     "docs/_static/drawing/example.html", pretty_print=True)

        .. raw:: html

            <iframe src="../_static/drawing/example.html"
            class="diagram-frame" height="500"></iframe>
        """
        from lxml.etree import SubElement, ElementTree
        from lxml.html import Element
        root = Element("div")
        style = SubElement(root, "style")
        style.text = TABLE_STYLE
        table = SubElement(root, "table")
        table.set("class", "diagram")
        width = self.max
        tr = SubElement(table, "tr")
        input_wires = self.rows[0]
        for i in range(width - 1):
            td = SubElement(tr, "td")
            if input_wires and input_wires[0].start - 1 == i:
                td.set("class", "wire")
                td.text = input_wires[0].label.name
                input_wires = input_wires[1:]
        for i, row in list(enumerate(self.rows))[1:]:
            tr = SubElement(table, "tr")
            if row and row[0].start > 0:
                td = SubElement(tr, "td")
                td.set("colspan", str(row[0].start))
            for cell, next_cell in zip(row, row[1:] + [None]):
                if cell.start == cell.stop:
                    td = SubElement(tr, "td")
                    td.set("class", "wire")
                    td.set("colspan", str(
                        width - cell.stop if next_cell is None
                        else next_cell.start - cell.stop))
                    if i == 0 or cell not in self.rows[i - 1]:
                        td.text = cell.label.name
                else:
                    td = SubElement(tr, "td")
                    td.set("class", "box")
                    td.set("colspan", str(cell.stop - cell.start))
                    td.text = cell.label.name
                    if cell.stop < width:
                        td = SubElement(tr, "td")
                        td.set("colspan", str(
                            width - cell.stop if next_cell is None
                            else next_cell.start - cell.stop))
        return ElementTree(root)

    def to_ascii(self, _debug=False) -> str:
        """
        Turn a grid into an ascii drawing.

        Examples
        --------
        >>> from discopy.monoidal import *
        >>> x = Ty('x')
        >>> f = Box('f', x, x @ x)
        >>> diagram = (f @ f[::-1] >> f @ f[::-1]).foliation()
        >>> print(diagram.to_grid().to_ascii())
         |         | |
         ---f---   -f-
         |     |   |
         -f-   --f--
         | |   |
        >>> cup, cap = Box('-', x @ x, Ty()), Box('-', Ty(), x @ x)
        >>> unit = Box('o', Ty(), x)
        >>> spiral = cap >> cap @ x @ x >> x @ x @ x @ unit @ x\\
        ...     >> x @ cap @ x @ x @ x @ x\\
        ...     >> x @ x @ unit[::-1] @ x @ x @ x @ x\\
        ...     >> x @ cup @ x @ x @ x >> x @ cup @ x >> cup
        >>> print(spiral.to_grid().to_ascii())
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
        space = "." if _debug else " "
        wire_chr, box_chr = "|", "-"

        def row_to_ascii(row):
            """ Turn a row into an ascii drawing. """
            result = ""
            for cell in row:
                result += (cell.start - len(result)) * space
                if isinstance(cell, Wire):
                    result += wire_chr
                else:
                    width = cell.stop - cell.start - 1
                    result += space + str(
                        cell.label.name)[:width].center(width, box_chr)
            return result

        return '\n'.join(map(row_to_ascii, self.rows)).strip('\n')

    @staticmethod
    def from_diagram(diagram: monoidal.Diagram) -> Grid:
        """
        Layout a diagram on a grid.

        The first row is a list of :class:`Wire` cells, then for each layer of
        the diagram there are two rows: the first for the boxes and the wires
        in between them, the second is a list of :class:`Wire` for the outputs.

        Parameters:
            diagram : The diagram to layout on a grid.

        >>> from discopy.monoidal import *
        >>> x = Ty('x')
        >>> f, s = Box('f', x, x @ x), Box('s', Ty(), Ty())
        >>> diagram = (
        ...     f @ f[::-1] >> x @ s @ x @ x >> f @ f[::-1]).foliation()
        >>> print(diagram.to_grid())
        Grid([Wire(1, x), Wire(15, x), Wire(17, x)],
             [Cell(0, 12, f), Cell(14, 18, f[::-1])],
             [Wire(1, x), Wire(11, x), Wire(15, x)],
             [Cell(0, 4, f), Cell(6, 8, s), Cell(10, 16, f[::-1])],
             [Wire(1, x), Wire(3, x), Wire(11, x)])
        """

        def make_boxes_as_small_as_possible(
                rows: list[list[Cell]]) -> list[list[Cell]]:
            for top, middle, bottom in zip(rows, rows[1:], rows[2:]):
                top_offset, bottom_offset = 0, 0
                for cell in middle:
                    if cell.start == cell.stop:
                        top_offset += 1
                        bottom_offset += 1
                    else:
                        box = cell.label
                        if not box.dom and not box.cod:
                            start, stop = cell.start, cell.start + 2
                        else:
                            start = min(
                                [top[top_offset + i].start
                                 for i, _ in enumerate(box.dom)] + [
                                    bottom[bottom_offset + i].start
                                    for i, _ in enumerate(box.cod)]) - 1
                            stop = max(
                                [top[top_offset + i].stop
                                 for i, _ in enumerate(box.dom)] + [
                                    bottom[bottom_offset + i].start
                                    for i, _ in enumerate(box.cod)]) + 1
                        cell.start, cell.stop = start, stop
                        top_offset += len(box.dom)
                        bottom_offset += len(box.cod)
            return rows

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
        result = Grid(make_boxes_as_small_as_possible(rows))
        return result - result.min

    def __add__(self, offset: int):
        return Grid([
            [cell + offset for cell in row] for row in self.rows])

    def __str__(self):
        rows_str = "],\n     [".join(map(
            lambda row: ", ".join(map(str, row)), self.rows))
        return f"Grid([{rows_str}])"

    __sub__ = Cell.__sub__
