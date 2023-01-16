
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
                        layer["a"] = np.array(a)
                        prev_out = layer_out
                
                        
                        
                
        


                        
        
