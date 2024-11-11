from discopy.interaction import *
from discopy.compact import Ty, Box, Swap, Cup, Cap
x0, y0, z0 = map(Ty, [obj + "0" for obj in "xyz"])
x1, y1, z1 = map(Ty, [obj + "1" for obj in "xyz"])
f, g = Box('f', x0 @ y1, x1 @ y0), Box('g', y0 @ z1, y1 @ z0)
caps = (x0 @ Cap(y1, y1.r) @ Cap(y0.r, y0) @ z1).foliation()
cups = (x1 @ Cup(y0, y0.r) @ Cup(y1.r, y1) @ z0).foliation()
symmetric_feedback = caps >> (f @ Swap(y1.r, y0.r) @ g).foliation() >> cups 
symmetric_feedback.draw()
