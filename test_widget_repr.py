import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from discopy.monoidal import Ty, Box

    x, y, z = Ty("x"), Ty("y"), Ty("z")
    diag = Box("f", x, y) >> Box("g", y, z) >> Box("h", z, x)
    return Box, Ty, diag


@app.cell
def _(diag):
    diag  # default SVG repr
    return


@app.cell
def _(diag):
    diag.to_widget()  # anywidget repr
    return


@app.cell
def _(diag):
    diag.to_typst()
    return


@app.cell
def _(Box, Ty):
    f = Box('f', Ty('x'), Ty('y'))

    doc = f.to_typst()
    doc
    return doc, f


@app.cell
def _(doc):
    source = doc.render()
    source
    return


@app.cell
def _(f):
    f.draw(format='typst')
    return


@app.cell
def _(Box):
    Box("$sum_(n=1)^oo 1/n^2 = pi^2/6$", 'x', 'y').draw(format='typst')
    return


@app.function
def draw_pregroup():
    from discopy.grammar.pregroup import Word, Cup, Ty
    n, s = Ty('n'), Ty('s')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ s @ Cup(n.l, n)
    return sentence.draw(format='typst', show=True)


@app.cell
def _():
    draw_pregroup()

    return


@app.cell
def _():
    def _():
        from discopy.rigid import Ty, Id, Cap, Cup
        from discopy.monoidal import Equation
        x = Ty('$Delta$')
        left  = x @ Cap(x.r, x) >> Cup(x, x.r) @ x
        right = Cap(x, x.l) @ x >> x @ Cup(x.l, x)
        eq = Equation(left, Id(x), right)
        eq.draw(format='typst', show=True)

    _()
    return


@app.cell
def _():
    def _():
        from discopy.frobenius import Ty, Spider, Equation, Box
        x = Ty('x')
        # special Frobenius: merging then splitting ≃ identity on a wire
        lhs = Spider(2, 1, x) >> Spider(1, 2, x)
        rhs = Spider(1, 1, x) @ Spider(1, 1, x)  # or just Id — pick your favourite law
        return (Spider(8, 3, x) >> (Spider(1, 2, x) @ (Spider(2, 1, x)>> Box("$Delta$", x, x)))).draw(format='typst', show=True)

    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
