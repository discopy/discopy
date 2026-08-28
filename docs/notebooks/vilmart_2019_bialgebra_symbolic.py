"""Exact checker for the bialgebra countermodel in the companion notebook.

Run with::

    uv run --with sympy python \
        docs/notebooks/vilmart_2019_bialgebra_symbolic.py

SymPy is kept as an on-demand dependency because the documentation notebook
itself uses only DisCoPy's normal NumPy dependency.
"""

import sympy as sp


def circle_remainder(expression, cosine, sine):
    """Reduce a polynomial modulo ``cosine ** 2 + sine ** 2 = 1``."""
    numerator = sp.together(expression)
    relation = sine ** 2 + cosine ** 2 - 1
    return sp.factor(sp.rem(
        sp.Poly(numerator, sine), sp.Poly(relation, sine)).as_expr())


def main():
    """Check the cubic scalar and the explicit violating component."""
    cosine, sine = sp.symbols("c s", real=True)
    dimension = 16
    weight = int.bit_count
    hadamard = sp.Matrix([
        [sp.Rational((-1) ** weight(x & y), 4)
         for y in range(dimension)]
        for x in range(dimension)])
    weight_two = [x for x in range(dimension) if weight(x) == 2]
    first = sp.Matrix([5, -1, -1, -1, -1, -1]) / sp.sqrt(30)
    second = (
        sp.Matrix([0, 4, -1, -1, -1, -1]) / (2 * sp.sqrt(5)))
    rotation = (
        sp.eye(6)
        + (cosine - 1) * (first * first.T + second * second.T)
        + sine * (second * first.T - first * second.T))
    change = sp.eye(dimension)
    for row, x in enumerate(weight_two):
        for column, y in enumerate(weight_two):
            change[x, y] = rotation[row, column]
    deformed = sp.simplify(change * hadamard * change.T)

    def reduce(expression):
        return circle_remainder(expression, cosine, sine)

    cubic = reduce(sum(
        deformed[x, y] ** 3
        for x in range(dimension)
        for y in range(dimension)))
    polynomial = (
        25 * cosine ** 5 + 25 * cosine ** 4 + 25 * cosine ** 3
        - 335 * cosine ** 2 - 398 * cosine - 110)

    def red_spider(first_input, second_input, output):
        return reduce(sum(
            deformed[first_input, j]
            * deformed[second_input, j]
            * deformed[output, j]
            for j in range(dimension)))

    witness_zero = red_spider(10, 10, 0)
    witness_fifteen = red_spider(10, 10, 15)
    expected = (
        -cosine ** 2 - 6 * cosine + 4 * sp.sqrt(6) * sine + 7
    ) / 96

    assert sp.factor(
        cubic - sp.Rational(1, 4)
        - (1 - cosine) * polynomial / 768) == 0
    assert witness_zero == sp.Rational(1, 4)
    assert sp.factor(witness_fifteen - expected) == 0
    assert polynomial.subs(cosine, -sp.Rational(1, 2)) == sp.Rational(
        93, 32)
    assert polynomial.subs(cosine, -sp.Rational(2, 5)) == -sp.Rational(
        702, 125)
    print(cubic)
    print(witness_zero)
    print(witness_fifteen)


if __name__ == "__main__":
    main()
