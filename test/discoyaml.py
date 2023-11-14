import yaml

from discopy.discoyaml import *


ob_x = from_yaml_documents(yaml.compose("x"))
box_x_to_y = from_yaml_documents(yaml.compose("x: y"))


def test_main():
    x, y, z = ob_x, Ob('y'), Ob('z')
    f, g, h = box_x_to_y, Box('g', y, z), Box('h', z, x)
    assert Id(x) >> f == f == f >> Id(y)
    assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
    assert f >> g >> h == f >> (g >> h)
    F = Functor(ob={x: y, y: z, z: x}, ar={f: g, g: h})
    assert F(Id(x)) == Id(F(x))
    assert F(f >> g) == F(f) >> F(g)
