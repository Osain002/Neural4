
import numpy as np
from .Activations import Activations


class dense_layers:

    def __init__(self):
        np.random.seed(1)
        self.dense_layers = []
        activations = {
            "sigmoid": (Activations.sigmoid, Activations.sigmoid_derivative),
            "relu": (Activations.relu, Activations.relu_derivative),
            "softmax": (Activations.softmax, Activations.softmax_derivative),
            "tanh": (Activations.tanh, Activations.tanh_derivative),
            "softplus": (Activations.softplus, Activations.sigmoid_derivative),
            "id": (Activations.id, Activations.id_der)
        }

    def update_input_size(self, new_in_size):
        next_layer_neurons = self.dense_layers[0]["num_neurons"]
        self.dense_layers[0]["weights"] = np.random.rand(
            new_in_size, next_layer_neurons)
        self.dense_layers[0]["in_size"] = new_in_size

    def dense_input_layer(self, input_size):
        input_layer = {
            "in_size": input_size,
            "activation": Activations["id"],
            "a": None,
            "deltas": None

        }

    def insert_dense(self, shape_layers, activation, **kwargs):
        input_size = None

        if "input_size" in kwargs.keys():
            input_size = kwargs["input_size"]

        for i in range(shape_layers[0]):
            if i == 0:
                weights = np.random.randn(input_size, shape_layers[1])
            else:
                weights = np.random.randn(shape_layers[1], shape_layers[1])

            bias = np.ones(shape_layers[1])

            layer = {
                "num_neurons": len(weights[0]),
                "weights": weights,
                "bias": bias,
                "a": None,
                "layer_in": None,
                "layer_out": None,
                "activation": self.activations[activation],
                "deltas": None
            }

            if i == 0:
                layer["in_size"] = input_size

            self.dense_layers.append(layer)

    def build_dense_layers(self, output_size, out_activation):
        prev_layer_out_size = self.dense_layers[-1]["num_neurons"]
        self.insert_dense(shape_layers=prev_layer_out_size, output_size=(
            1, output_size), activation=out_activation)
