# -*- coding: utf-8 -*-

"""
"""

from discopy import cat, config, messages, monoidal, rigid
from discopy.monoidal import PRO
import tensorflow as tf
from tensorflow import keras


class Network(monoidal.Box):
    """ Implements tensorflow neural networks

    >>> a = Network.dense_model(16, 12, [5, 6])
    >>> b = Network.dense_model(12, 2, [5])
    >>> assert (a >> b).model.layers[1:] == a.model.layers[1:] + b.model.layers[1:]
    >>> assert (a >> Network.id(12)).model == a.model
    """

    def __init__(self, dom, cod, model):
        self.model = model
        super().__init__("Network", dom, cod)

    def then(self, other):
        inputs = keras.Input(shape=(len(self.dom),))
        output = self.model(inputs)
        output = other.model(output)
        composition = keras.Model(inputs=inputs, outputs=output)
        return Network(self.dom, other.cod, composition)

    def tensor(self, other):
        dom = len(self.dom) + len(other.dom)
        cod = len(self.cod) + len(other.cod)
        inputs = keras.Input(shape=(dom,))
        model1 = keras.layers.Lambda(
            lambda x: x[:, :len(self.dom)],)(inputs)
        model2 = keras.layers.Lambda(
            lambda x: x[:, len(self.dom):],)(inputs)
        model1 = self.model(model1)
        model2 = other.model(model2)
        outputs = keras.layers.Concatenate()([model1, model2])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return Network(PRO(dom), PRO(cod), model)

    @staticmethod
    def id(dim):
        inputs = keras.Input(shape=(len(dim),))
        return Network(dim, dim, keras.Model(inputs=inputs, outputs=inputs))

    @staticmethod
    def dense_model(dom, cod, hidden_layer_dims=[], activation=tf.nn.relu):
        inputs = keras.Input(shape=(dom,))
        model = inputs
        for dim in hidden_layer_dims:
            model = keras.layers.Dense(dim, activation=activation)(model)
        outputs = keras.layers.Dense(cod, activation=activation)(model)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return Network(PRO(dom), PRO(cod), model)