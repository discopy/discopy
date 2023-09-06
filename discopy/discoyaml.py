import yaml

from discopy import frobenius


class Ob(frobenius.Ob):
    """"""

class Ty(frobenius.Ty):
    """"""

class MappingBox(frobenius.Box):
    """"""

class Box(MappingBox):
    """"""

def from_yaml(data: yaml.Node) -> Box:
    match data:
        case yaml.ScalarNode(value=value):
            return frobenius.Ty(value)
        case yaml.MappingNode(value=[(key_node, value_node)]):
            return frobenius.Box("f", from_yaml(key_node), from_yaml(value_node))
