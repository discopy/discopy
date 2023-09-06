import yaml

from discopy import frobenius


class Ty(frobenius.Ty):
    """"""

class MappingBox(frobenius.Box):
    """"""

class SequenceBox(frobenius.Box):
    """"""

class CollectionBox(MappingBox, SequenceBox):
    """"""

class Box(Ty, CollectionBox):
    """"""

def from_yaml(data: yaml.Node) -> Box:
    match data:
        case yaml.ScalarNode(value=value):
            return Ty(value)
