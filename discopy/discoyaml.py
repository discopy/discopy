import yaml
from yaml import ScalarNode

from discopy.cat import Ob
from discopy.frobenius import Diagram

def from_yaml(data: yaml.Node) -> Diagram:
    match data:
        case ScalarNode(value=value):
            return Ob(value)
