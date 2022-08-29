import discopy.cat as cat


def distribute_composition_cat(
    arrow: cat.Arrow,
    index_sum: int,
    distribute_up_to: int,
    arrow_construction=lambda dom, cod, boxes: cat.Arrow(dom, cod, boxes),
) -> cat.Arrow:
    """
    Distributes the sum at index_sum to the left with

    Parameters
    ----------
    arrow: Arrow
        The arrow to perform the operation on
    index_sum: int
        The index of the sum to distribute for
    distribute_up_to: int
        index of the last term to distribute
    arrow_construction: callable
        callable returning the arrow equivalent in the relevant category

    Raises
    ------
      IndexError: The index of sum does not correspond to a box
                  or distribute_up_to is negative or to large
      TypeError: The box at index_sum does not have type sum
    """
    if len(arrow.boxes) <= index_sum or index_sum < 0:
        raise IndexError("index to large or negative, no such box")
    if (
        len(arrow.boxes) <= distribute_up_to
        or distribute_up_to < 0
        or distribute_up_to == index_sum
    ):
        raise IndexError(
            "distribute_up_to to large, negative or \
equal to index_sum, no such box"
        )
    if not isinstance(arrow.boxes[index_sum], cat.LocalSum):
        raise TypeError("box at index %d is not a LocalSum", index_sum)

    if index_sum < distribute_up_to:
        # distribute to the right
        unit = arrow.boxes[index_sum].__class__(
            [], arrow.boxes[index_sum].dom, arrow.boxes[distribute_up_to].cod
        )
        terms = [
            g.then(
                arrow_construction(
                    arrow.boxes[index_sum + 1].dom,
                    arrow.boxes[distribute_up_to].cod,
                    arrow.boxes[index_sum + 1: distribute_up_to + 1],
                )
            )
            for g in arrow.boxes[index_sum].terms
        ]
        term = arrow.boxes[index_sum].upgrade(cat.LocalSum(terms))
        new_boxes = (
            arrow.boxes[:index_sum] + [term]
            + arrow.boxes[distribute_up_to + 1:]
        )
        return arrow_construction(
            new_boxes[0].dom,
            new_boxes[-1].cod,
            new_boxes,
        )
    else:
        # distribute to the left
        unit = arrow.boxes[index_sum].__class__(
            [], arrow.boxes[distribute_up_to].dom, arrow.boxes[index_sum].cod
        )
        terms = [
            arrow_construction(
                arrow.boxes[distribute_up_to].dom,
                arrow.boxes[index_sum - 1].cod,
                arrow.boxes[distribute_up_to:index_sum],
            ).then(g)
            for g in arrow.boxes[index_sum].terms
        ]
        term = arrow.boxes[index_sum].upgrade(cat.LocalSum(terms))
        new_boxes = (
            arrow.boxes[:distribute_up_to]
            + [term] + arrow.boxes[index_sum + 1:]
        )
        return arrow_construction(
            new_boxes[0].dom,
            new_boxes[-1].cod,
            new_boxes,
        )


