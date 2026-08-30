"""Exact checker for the bialgebra countermodel in the companion notebook.

Run with::

    uv run --with sympy python \
        docs/notebooks/vilmart_2019_bialgebra_symbolic.py

SymPy is kept as an on-demand dependency because the documentation notebook
itself uses only DisCoPy's normal NumPy dependency.
"""

from itertools import product

import sympy as sp


def circle_remainder(expression, cosine, sine):
    """Reduce a polynomial modulo ``cosine ** 2 + sine ** 2 = 1``."""
    numerator = sp.together(expression)
    relation = sine ** 2 + cosine ** 2 - 1
    return sp.factor(sp.rem(
        sp.Poly(numerator, sine), sp.Poly(relation, sine)).as_expr())


def assert_bialgebra(matrix):
    """Check every component of the normalized bialgebra equation."""
    dimension = matrix.rows
    indices = range(dimension)
    tensor = {
        (first, second, output): sp.simplify(sum(
            matrix[first, middle]
            * matrix[second, middle]
            * matrix[output, middle]
            for middle in indices))
        for first, second, output in product(indices, repeat=3)}
    for first, second, output, other_output in product(indices, repeat=4):
        entry = tensor[first, second, output]
        left = (
            sp.sqrt(dimension)
            * entry
            * tensor[first, second, other_output])
        right = int(output == other_output) * entry
        assert sp.simplify(left - right) == 0


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
    polynomial_in_cosine = sp.Poly(polynomial, cosine)
    assert sp.gcd(
        polynomial_in_cosine,
        sp.Poly(sp.diff(polynomial, cosine), cosine)) == 1

    lower_bound = -sp.Rational(1, 2)
    upper_bound = -sp.Rational(2, 5)
    root_interval = sp.Interval.open(lower_bound, upper_bound)
    assert polynomial_in_cosine.count_roots(
        lower_bound, upper_bound) == 1
    admissible_roots = [
        root for root in polynomial_in_cosine.all_roots()
        if root.is_real and root_interval.contains(root) is sp.true]
    assert len(admissible_roots) == 1
    admissible_cosine, = admissible_roots
    assert sp.simplify(polynomial.subs(
        cosine, admissible_cosine)) == 0
    admissible_sine = sp.sqrt(1 - admissible_cosine ** 2)
    assert admissible_sine.is_positive is True
    root_substitution = {
        cosine: admissible_cosine, sine: admissible_sine}
    assert sp.simplify(
        cubic.subs(root_substitution) - sp.Rational(1, 4)) == 0
    witness_at_root = expected.subs(root_substitution)
    # This is an exact RootOf sign decision, not a floating-point check.
    assert (
        witness_at_root - sp.Rational(61, 640)).is_positive is True

    # Classify the connected N=3 phase-central family.  The Walsh transform
    # exchanges the weight-one and weight-two planes, so two block rotations
    # enter only through the sum of their angles.  It is enough to rotate the
    # weight-one plane by that sum.
    dimension_three = 8
    hadamard_three = sp.Matrix([
        [(-1) ** weight(x & y) * sp.sqrt(2) / 4
         for y in range(dimension_three)]
        for x in range(dimension_three)])
    uniform_three = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    first_three = sp.Matrix([1, -1, 0]) / sp.sqrt(2)
    second_three = sp.Matrix([1, 1, -2]) / sp.sqrt(6)
    rotation_three = (
        uniform_three * uniform_three.T
        + cosine * (
            first_three * first_three.T
            + second_three * second_three.T)
        + sine * (
            second_three * first_three.T
            - first_three * second_three.T))
    change_three = sp.eye(dimension_three)
    weight_one = [x for x in range(dimension_three) if weight(x) == 1]
    for row, x in enumerate(weight_one):
        for column, y in enumerate(weight_one):
            change_three[x, y] = rotation_three[row, column]
    deformed_three = sp.simplify(
        change_three * hadamard_three * change_three.T)
    cubic_three = reduce(sum(
        deformed_three[x, y] ** 3
        for x in range(dimension_three)
        for y in range(dimension_three)))
    residual_three = (
        sp.sqrt(2) * (cosine - 1) * (2 * cosine + 1) ** 2 / 3)
    assert sp.simplify(
        cubic_three - sp.sqrt(2) / 4 - residual_three) == 0

    # The non-identity EU roots give Walsh matrices and still satisfy B.
    for sine_sign in (-1, 1):
        root_three = sp.simplify(deformed_three.subs({
            cosine: -sp.Rational(1, 2),
            sine: sine_sign * sp.sqrt(3) / 2}))
        assert set(root_three) == {-sp.sqrt(2) / 4, sp.sqrt(2) / 4}
        assert_bialgebra(root_three)

    print(cubic)
    print(witness_zero)
    print(witness_fifteen)


if __name__ == "__main__":
    main()
