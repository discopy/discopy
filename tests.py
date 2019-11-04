
a, b, c = Ob('a'), Ob('a'), 'c'
assert a == b and b != c

x, y, z = Ob('x'), Ob('y'), Ob('z')
f, g, h = Generator('f', x, y), Generator('g', y, z), Generator('h', z, x)
assert Arrow.id(x).then(f) == f == f.then(Arrow.id(y))
assert (f.then(g)).dom == f.dom and (f.then(g)).cod == g.cod
assert f.then(g).then(h) == f.then(g.then(h)) == Arrow(x, x, [f, g, h])

a = f.then(g).then(h)
F = Functor({x: int, y:tuple, z:int}, {
    f: Function(lambda x: (x, x), int, tuple),
    g: Function(lambda x: x[0] + x[1], tuple, int),
    h: Function(lambda x: x // 2, int, int)})
# bigF is a functor from the free category to Cat, i.e. it maps f to F
bigF = Functor({x: Arrow, y: Arrow}, {f: Function(F, Arrow, Arrow)})

assert F(Arrow.id(x))(SEED) == Function(lambda x: x, int, int)(SEED) == SEED
assert F(f.then(g))(SEED) == F(g)(F(f)(SEED))
assert F(a)(SEED) == F(h)(F(g)(F(f)(SEED))) == F(Arrow.id(x))(SEED) == SEED
assert isinstance(bigF(f).name, Functor)
assert bigF(f)(f)(SEED) == F(f)(SEED) == (SEED, SEED)
x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
f0, f1 = Box('f0', x, y), Box('f1', z, w)
assert (f0 @ f1).interchange(0, 1) == Id(x) @ f1 >> f0 @ Id(w)
assert (f0 @ f1).interchange(0, 1).interchange(0, 1) == f0 @ f1

s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)

assert x + y != y + x
assert (x + y) + z == x + y + z == x + (y + z) == sum([x, y, z], Ty())

x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
f, g, h = Box('f', x, x + y), Box('g', y + z, w), Box('h', x + w, x)
d = Id(x) @ g << f @ Id(z)

F0 = NumpyFunctor({x: 1, y: 2, z: 3, w: 4}, dict())
F = NumpyFunctor({x: 1, y: 2, z: 3, w: 4},
                 {a: np.zeros(F0(a.dom) + F0(a.cod)) for a in [f, g, h]})

assert F(d.dagger()).shape == tuple(F(d.cod) + F(d.dom))

s, n = Pregroup('s'), Pregroup('n')

Alice, Bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r + s + n.l)
grammar = Cup(n) @ Wire(s) @ Cup(n.l)
sentence = grammar << Alice @ loves @ Bob
assert sentence == Parse([Alice, loves, Bob], [0, 1]).interchange(0, 1)\
                                                     .interchange(1, 2)\
                                                     .interchange(0, 1)
F = Model({s: 1, n: 2},
          {Alice: [1, 0],
           loves: [0, 1, 1, 0],
           Bob: [0, 1]})
assert F(sentence) == True

snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)
assert (F(snake_l) == F(Wire(n))).all()
assert (F(Wire(n)) == F(snake_r)).all()
