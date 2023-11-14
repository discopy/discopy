import yaml

from discopy.cat import factory
from discopy import frobenius


class Ob(frobenius.Ob):
    """"""

@factory
class Ty(frobenius.Ty):
    """"""
    ob_factory = Ob

@factory
class Diagram(frobenius.Diagram):
    """"""
    ob_factory = Ob
    ty_factory = Ty

class Box(frobenius.Box, Diagram):
    """"""

class Category(frobenius.Category):
    """"""
    ob, ar = Ty, Diagram

class Functor(frobenius.Functor):
    """"""
    dom = cod = Category()

def from_yaml_documents(*nodes: yaml.Node) -> Box:
    for node in nodes:
        return from_yaml_node(node)

def from_yaml_node(data: yaml.Node) -> Box:
    match data:
        case yaml.ScalarNode(value=value):
            return Ty(value)
        case yaml.MappingNode(value=[(key_node, value_node)]):
            return Box("f", from_yaml_node(key_node), from_yaml_node(value_node))

Id = Diagram.id
