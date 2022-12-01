import pickle

from discopy import closed


def test_tree2diagram():
    tree, boxes, offsets, rigid_boxes, rigid_offsets =\
        pickle.load(open("test/src/tree2diagram.pickle", "rb"))
    diagram = tree2diagram(tree)
    rigid_diagram = closed.to_rigid(diagram)
    assert diagram.boxes == boxes
    assert diagram.offsets == offsets
    assert rigid_diagram.boxes == rigid_boxes
    assert rigid_diagram.offsets == rigid_offsets
