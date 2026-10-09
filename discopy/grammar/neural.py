# -*- coding: utf-8 -*-

"""
Neural parsing of abstract categorial grammars into proof nets, with PyTorch.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Signature
    Tagger
    Linker
    Parser

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    sinkhorn

Parsing in two steps
--------------------

A parser goes from words to a :class:`discopy.grammar.proofnet.ProofNet` in
two steps, the two halves of a proof net: the sequent, then the linking
(Kogkalidis, Moortgat and Moot, `Neural proof nets (2020)
<https://aclanthology.org/2020.conll-1.3/>`_). Both read the states of the
words given by any encoder, a tensor ``(batch, length, hidden)`` with a
boolean mask of the same batch and length saying which states are words, so
that a parser can be trained jointly with the encoder by summing its losses.

The :class:`Tagger` gives each word a type. It is constructive: rather than
picking a type out of the ones seen in training, it writes one symbol at a
time in Polish notation, the :class:`Signature` of the grammar, and it only
writes symbols that can be completed into a type, so that every output is a
type and it can output types it has never seen (Kogkalidis, Moortgat and
Deoskar, `Constructive type-logical supertagging with self-attention networks
(2019) <https://aclanthology.org/W19-4314/>`_).

The :class:`Linker` scores each pair of a negative and a positive port, from
the states of their words and what they are, and
:meth:`discopy.grammar.proofnet.Sequent.decode` takes the best linking. A
linking is a permutation of each block of ports, and a permutation matrix
is a vertex of the polytope of doubly stochastic matrices: :func:`sinkhorn`
normalises scores to the nearest such matrix in entropy, and the linker is
trained on the likelihood of the gold permutation there (Mena et al.,
`Learning latent permutations with Gumbel-Sinkhorn networks (2018)
<https://arxiv.org/abs/1802.08665>`_). The best linking does not depend on
the normalisation, so that decoding takes the raw scores.

Example
-------
>>> import torch
>>> from discopy.grammar.abstract import Ty
>>> from discopy.grammar.proofnet import ProofNet
>>> n, s = Ty("n"), Ty("s")
>>> Alice, loves, Bob = n("Alice"), (n >> (n >> s))("loves"), n("Bob")
>>> nets = [ProofNet.from_term(loves(y)(x), (x, loves, y))
...         for x, y in [(Alice, Bob), (Bob, Alice)]]
>>> _ = torch.manual_seed(0)
>>> parser = Parser(Signature.from_types([n, s]), hidden=8, dim=8)
>>> encoder = torch.nn.Embedding(3, 8)
>>> sentences = torch.tensor([[0, 1, 2], [2, 1, 0]])
>>> mask = sentences >= 0
>>> optimiser = torch.optim.Adam(
...     [*parser.parameters(), *encoder.parameters()], lr=.05)
>>> for _ in range(100):
...     optimiser.zero_grad()
...     parser.loss(encoder(sentences), mask, nets).backward()
...     optimiser.step()
>>> states = encoder(sentences)[1]
>>> net = parser.eval()(["Bob", "loves", "Alice"], states, s)
>>> print(net.to_term())
(n >> (n >> s))('loves')(n('Alice'))(n('Bob'))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # pylint: disable=import-error  # torch is an extra
from torch import nn  # pylint: disable=import-error  # torch is an extra

from discopy.grammar.abstract import Ty
from discopy.grammar.proofnet import Sequent, ProofNet, ports


@dataclass(frozen=True)
class Signature:
    """
    The symbols of the types of a grammar in Polish notation: the
    exponential, of arity two, then the atoms, of arity zero.

    Parameters:
        atoms : The atomic types.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> signature = Signature((x, y, z))
    >>> signature.encode((x >> y) >> z)
    [0, 0, 1, 2, 3]
    >>> print(signature.decode([0, 0, 1, 2, 3]))
    ((x >> y) >> z)
    """
    atoms: tuple[Ty, ...]

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(self.atoms))

    def __len__(self):
        return 1 + len(self.atoms)

    @property
    def arities(self) -> list[int]:
        "The number of arguments of each symbol."
        return [2] + [0] * len(self.atoms)

    @classmethod
    def from_types(cls, types) -> Signature:
        "The signature of the atoms of some types, in order of occurrence."
        return cls(tuple(dict.fromkeys(
            atom for ty in types for atom, _ in ports(ty))))

    def encode(self, ty: Ty) -> list[int]:
        "The symbols of a type, the exponent before the base."
        if ty.is_exp:
            return [0] + self.encode(ty.exponent) + self.encode(ty.base)
        return [1 + self.atoms.index(ty)]

    def decode(self, symbols) -> Ty:
        "The type of some symbols, the inverse of :meth:`encode`."
        symbols = iter(symbols)

        def tree():
            symbol = next(symbols)
            if symbol:
                return self.atoms[symbol - 1]
            exponent = tree()
            return exponent >> tree()
        return tree()


def words(states, mask):
    """
    The state of each word, with the states and the mask of its sentence.

    Parameters:
        states : The states of a batch, ``(batch, length, hidden)``.
        mask : Whether each state is a word, ``(batch, length)``.
    """
    sentence = mask.nonzero()[:, 0]
    return states[mask], states[sentence], mask[sentence]


class Tagger(nn.Module):
    """
    A constructive supertagger: a recurrent decoder writing the type of each
    word in Polish notation, starting from the state of the word and
    attending to the states of its sentence.

    Parameters:
        signature : The symbols of the types.
        hidden : The size of the states.
        dim : The size of the embedding of each symbol.
        dropout : The probability of dropping a feature in training.
    """
    def __init__(self, signature: Signature, hidden: int, dim: int = 256,
                 dropout: float = 0.):
        super().__init__()
        self.signature = signature
        self.embedding = nn.Embedding(len(signature) + 1, dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.query = nn.Linear(hidden, hidden)
        self.output = nn.Linear(2 * hidden, len(signature))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "arities", torch.tensor(signature.arities), persistent=False)

    def forward(self, symbols, hidden, keys, mask):
        """
        The scores of the next symbol after each of some symbols, and the
        hidden state after the last one.

        Parameters:
            symbols : The symbols written so far, ``(words, steps)``,
                ``len(signature)`` standing for the start.
            hidden : The hidden state before them, ``(1, words, hidden)``.
            keys : The states of the sentence of each word.
            mask : Whether each of these states is a word.
        """
        outputs, hidden = self.gru(
            self.dropout(self.embedding(symbols)), hidden)
        attention = torch.einsum("wsh,wkh->wsk", self.query(outputs), keys)\
            / keys.shape[-1] ** .5
        attention = attention.masked_fill(~mask[:, None], -torch.inf)
        context = torch.einsum("wsk,wkh->wsh", attention.softmax(-1), keys)
        return self.output(self.dropout(
            torch.cat([outputs, context], -1))), hidden

    def loss(self, states, mask, types) -> torch.Tensor:
        """
        The mean negative log-likelihood of each symbol of the types.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            types : The type of each word, in order.
        """
        hidden, keys, key_mask = words(states, mask)
        targets = nn.utils.rnn.pad_sequence([
            torch.tensor(self.signature.encode(ty)) for ty in types],
            batch_first=True, padding_value=-1).to(states.device)
        symbols = torch.cat([
            torch.full_like(targets[:, :1], -1), targets[:, :-1]], 1)
        symbols = symbols.masked_fill(symbols < 0, len(self.signature))
        scores, _ = self(symbols, hidden[None], keys, key_mask)
        return nn.functional.cross_entropy(
            scores.flatten(0, 1), targets.flatten(), ignore_index=-1)

    @torch.no_grad()
    def beam(self, states, mask, k: int = 1, max_length: int = 32
             ) -> list[list[tuple[Ty, float]]]:
        """
        The ``k`` most likely types of each word with their log-probability,
        by beam search over the symbols that can still be completed into a
        type of at most ``max_length`` symbols.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            k : The number of types of each word.
            max_length : The largest number of symbols of a type.
        """
        hidden, keys, key_mask = (
            x.repeat_interleave(k, 0) for x in words(states, mask))
        n_words, start = len(hidden) // k, len(self.signature)
        arities = torch.cat([self.arities, self.arities.new_ones(1)])
        scores = torch.full((n_words, k), -torch.inf, device=states.device)
        scores[:, 0] = 0
        slots = torch.ones_like(scores, dtype=torch.long).flatten()
        history = slots.new_empty((n_words * k, 0))
        symbols, hidden = slots.new_full((n_words * k, 1), start), hidden[None]
        for step in range(max_length):
            if not slots[scores.flatten() > -torch.inf].any():
                break
            logits, hidden = self(symbols, hidden, keys, key_mask)
            allowed = slots[:, None] + arities[None, :-1] <= max_length - step
            log_p = logits[:, -1].masked_fill(~allowed, -torch.inf)\
                .log_softmax(-1).masked_fill(slots[:, None] == 0, -torch.inf)
            log_p = torch.cat([log_p, torch.where(
                slots[:, None] == 0, 0., -torch.inf)], 1)
            scores, best = (scores.flatten()[:, None] + log_p)\
                .view(n_words, -1).topk(k)
            beams = (best // (start + 1)
                     + torch.arange(n_words, device=best.device)[:, None] * k
                     ).flatten()
            symbols = (best % (start + 1)).flatten()[:, None]
            hidden, history = hidden[:, beams], torch.cat(
                [history[beams], symbols], 1)
            slots = slots[beams] + arities[symbols[:, 0]] - 1
        return [[(self.signature.decode(row[row < start].tolist()), score)
                 for row, score in zip(rows, beam.tolist())
                 if score > -torch.inf]
                for rows, beam in zip(history.view(n_words, k, -1), scores)]


def sinkhorn(blocks, iterations: int = 10) -> torch.Tensor:
    """
    The log-domain Sinkhorn normalisation of square blocks of scores,
    alternately normalising rows and columns towards a doubly stochastic
    matrix (Sinkhorn and Knopp, `Concerning nonnegative matrices and doubly
    stochastic matrices (1967) <https://doi.org/10.2140/pjm.1967.21.343>`_).

    Parameters:
        blocks : Square matrices of scores.
        iterations : The number of normalisations of rows and columns.

    Returns:
        The log-probabilities, padded into one tensor ``(blocks, n, n)``
        where each padded row only reaches its padded column.

    Example
    -------
    >>> log_p = sinkhorn([torch.randn(2, 2), torch.randn(3, 3)], 100)
    >>> assert torch.allclose(log_p.exp().sum(1), torch.ones(2, 3))
    >>> assert torch.allclose(log_p.exp().sum(2), torch.ones(2, 3))
    """
    n = max(len(block) for block in blocks)
    sizes = torch.tensor([len(block) for block in blocks])
    real = torch.arange(n)[None] < sizes[:, None]
    real = (real[:, :, None] & real[:, None]).to(blocks[0].device)
    padding = torch.eye(n, dtype=torch.bool, device=real.device) & ~real
    scores = torch.stack([
        nn.functional.pad(block, (0, n - len(block), 0, n - len(block)))
        for block in blocks]).masked_fill(~real, -torch.inf)
    scores = scores.masked_fill(padding, 0.)
    for _ in range(iterations):
        scores = scores - scores.logsumexp(2, keepdim=True)
        scores = scores - scores.logsumexp(1, keepdim=True)
    return scores


class Linker(nn.Module):
    """
    A bilinear scorer of the pairs of ports of a sequent: each port is
    embedded from the state of its word, the mean state for the goal, with
    its atom, its polarity and its index within the type of its word, which
    tells apart the subject and the object of a verb.

    Parameters:
        signature : The atoms of the types.
        hidden : The size of the states.
        dim : The size of the embedding of each port.
        max_index : The number of indices within a type told apart.
        iterations : The number of :func:`sinkhorn` normalisations.
    """
    def __init__(self, signature: Signature, hidden: int, dim: int = 256,
                 max_index: int = 32, iterations: int = 10):
        super().__init__()
        self.signature, self.iterations = signature, iterations
        self.atom = nn.Embedding(len(signature.atoms), dim)
        self.polarity = nn.Embedding(2, dim)
        self.index = nn.Embedding(max_index, dim)
        self.port = nn.Linear(hidden + dim, dim)
        self.bilinear = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(dim, dim)))

    def forward(self, states, sequent: Sequent) -> torch.Tensor:
        """
        The score of each pair of ports, a square matrix over the ports.

        Parameters:
            states : The states of the words of the sequent.
            sequent : The words with their types and the goal.
        """
        owners = [(i, word.cod, True) for i, word in enumerate(sequent.words)]
        features = torch.tensor([
            (i, self.signature.atoms.index(atom), polarity, j)
            for i, ty, given in owners + [(len(owners), sequent.goal, False)]
            for j, (atom, polarity) in enumerate(ports(ty, given))],
            device=states.device)
        states = torch.cat([states, states.mean(0, keepdim=True)])
        rows, atoms, polarities, indices = features.T
        embedding = self.atom(atoms) + self.polarity(polarities)\
            + self.index(indices.clamp(max=self.index.num_embeddings - 1))
        ports_ = torch.tanh(self.port(torch.cat([states[rows], embedding], 1)))
        return ports_ @ self.bilinear @ ports_.T / len(self.bilinear) ** .5

    def loss(self, states, mask, nets) -> torch.Tensor:
        """
        The mean negative log-likelihood of each link of the nets, after
        :func:`sinkhorn` normalisation of each block.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            nets : The proof net of each sentence.
        """
        blocks, gold = [], []
        for sentence, real, net in zip(states, mask, nets):
            scores = self(sentence[real], net.sequent)
            partner = dict(net.links)
            for negatives, positives in net.sequent.blocks.values():
                blocks.append(scores[negatives][:, positives])
                gold.append(torch.tensor(
                    [positives.index(partner[i]) for i in negatives]))
        log_p = sinkhorn(blocks, self.iterations)
        return -torch.cat([
            block[torch.arange(len(target)), target]
            for block, target in zip(log_p, gold)]).mean()


class Parser(nn.Module):
    """
    A parser into proof nets, a :class:`Tagger` and a :class:`Linker` over
    the states of the same encoder, called on a sentence with its states and
    its goal.

    Parameters:
        signature : The symbols of the types.
        hidden : The size of the states.
        dim : The size of the embeddings of the tagger and of the linker.
    """
    def __init__(self, signature: Signature, hidden: int, dim: int = 256):
        super().__init__()
        self.tagger = Tagger(signature, hidden, dim)
        self.linker = Linker(signature, hidden, dim)

    def loss(self, states, mask, nets) -> torch.Tensor:
        """
        The sum of the losses of the tagger and of the linker.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            nets : The proof net of each sentence, its words in order.
        """
        types = [word.cod for net in nets for word in net.sequent.words]
        return self.tagger.loss(states, mask, types)\
            + self.linker.loss(states, mask, nets)

    @torch.no_grad()
    def forward(self, sentence: list[str], states, goal: Ty) -> ProofNet:
        """
        The proof net of the most likely types and the best linking.

        Parameters:
            sentence : The words.
            states : Their states, ``(length, hidden)``.
            goal : The type of the sentence.
        """
        mask = states.new_ones(1, len(states), dtype=torch.bool)
        types = [beam[0][0] for beam in self.tagger.beam(states[None], mask)]
        sequent = Sequent(
            tuple(ty(word) for word, ty in zip(sentence, types)), goal)
        return sequent.decode(self.linker(states, sequent).cpu())
