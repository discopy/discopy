from discopy.tensor import Box, Dim, Id, Cup


alice = Box("Alice", Dim(1), Dim(2), data=[1, 2])
eats = Box("eats", Dim(1), Dim(2, 3, 2), data=[3] * 12)
food = Box("food", Dim(1), Dim(2), data=[4, 5])

pick = alice @ eats @ food >>\
          Cup(Dim(2), Dim(2)) @ Id(Dim(3)) @ Cup(Dim(2), Dim(2))
