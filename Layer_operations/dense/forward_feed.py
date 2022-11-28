
import numpy as np


class feed_forward:
        def __init__(self, dense_layers):
                self.layers = dense_layers


        def feed_forward_dense(self, input):
                prev_out = input
                layers = self.layers
                for i, layer in enumerate(layers, 1):
                        layer["input"] = prev_out
                        weights = layer['weights']
                        bias = layer['bias']
                        activation = layer["activation"][0]
                        a_ = np.matmul(weights.T , prev_out)
                        a = np.add(a_, bias )
                        layer_out = activation(a)
                        layer["layer_out"] = layer_out
                        layer["a"] = a
                        prev_out = layer_out
                


        # def calculate_layer_output(input, layer):
        #         layer_weights = layer["weights"]
        #         layer_activation = layer["activation"]

        #         a = np.matmul(layer_weights.T, input)
        #         a = np.add(a, layer["bias"])

        #         layer_out = layer_activation[0](a)
        #         layer["a"] = a
        #         layer["layer_out"] = layer_out




                # return layer_out

        
        # def run_forward_feed(self, dense_layers, input):
        #         print("gg")
        #         for i, layer in enumerate(dense_layers):
        #                 layer["input"] = input
        #                 input = feed_forward.calculate_layer_output(input, layer)
                        
                        
                
        


                        
        
