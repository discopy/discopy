from discopy.tensor import Box, Dim, Id, Cup


alice = Box[int]("Alice", Dim(1), Dim(2), [1, 2])
eats = Box[int]("eats", Dim(1), Dim(2, 3, 2), [3] * 12)
food = Box[int]("food", Dim(1), Dim(2), [4, 5])

pick = alice @ eats @ food >>\
          Cup(Dim(2), Dim(2)) @ Id(Dim(3)) @ Cup(Dim(2), Dim(2))
