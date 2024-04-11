from discopy.quantum.circuit import Qudit, Ty


pick = [Qudit(d) for d in range(2, 5)]
pick += list(map(Ty, pick))
