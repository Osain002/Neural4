
import numpy as np
from .Activations import Activations


class dense_layers:

        def __init__(self):
                np.random.seed(1)
                self.dense_layers = []
                

        def update_input_size(self, new_in_size):
                next_layer_neurons = self.dense_layers[0]["num_neurons"]
                self.dense_layers[0]["weights"] = np.random.rand(new_in_size, next_layer_neurons)
                self.dense_layers[0]["in_size"] = new_in_size



        def insert_dense(self, input_size, shape_layers, activation):
                activations = {
                        "sigmoid":( Activations.sigmoid, Activations.sigmoid_derivative),
                        "relu": (Activations.relu, Activations.relu_derivative),
                        "softmax": (Activations.softmax, Activations.softmax_derivative),
                        "tanh": (Activations.tanh, Activations.tanh_derivative)
                }


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
                                "layer_in":None,
                                "layer_out": None,
                                "activation": activations[activation],
                                "deltas": None
                        }

                        if i == 0:
                                layer["in_size"] = input_size
                
                        self.dense_layers.append(layer)
                
        
        def build_dense_layers(self, output_size, out_activation):
                prev_layer_out_size = self.dense_layers[-1]["num_neurons"]
                self.insert_dense(prev_layer_out_size, (1, output_size), out_activation)
                
                
                

