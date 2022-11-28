import numpy as np
from .Activations import Activations

class ConvLayers:
        def __init__(self):
                self.convNet_layers = []
                self.activations = {
                        "sigmoid":( Activations.sigmoid, Activations.sigmoid_derivative),
                        "relu": (Activations.relu, Activations.relu_derivative),
                        "softmax": (Activations.softmax, Activations.softmax_derivative),
                        "tanh": (Activations.tanh, Activations.tanh_derivative),
                        "None": (None, None)
                }
        


        def add_convolution_layer(self, num_filters, filter_size, activation):


                if activation not in self.activations.keys():
                        raise Exception("Invalid activation")
                else:
                        layer = {
                                "layer_num": len(self.convNet_layers),
                                "layer_type": "conv",
                                "filter_size": filter_size,
                                "de_pool_matrix": [],
                                "filters": [np.random.rand(filter_size, filter_size) for i in range(num_filters)],
                                "bias": [], #this will be initialised in build_network
                                "activation": self.activations[activation][0],
                                "activation_derivative": self.activations[activation][1],
                                "stride": 1
                        }

                        self.convNet_layers.append(layer)

        def add_pooling_layer(self, type, kernal_size, stride):
                pool_functions = {
                        "max": np.max, 
                        "average": np.mean
                }

                if type not in pool_functions.keys():
                        raise Exception("Invalid pooling type")
                else:
                        pool = {
                                "layer_num": len(self.convNet_layers),
                                "layer_type": "pooling",
                                "de_pool_matrix": [],
                                "pool_func": pool_functions[type],
                                "ker_size": kernal_size,
                                "stride": stride
                        }

                        self.convNet_layers.append(pool)
        
        def build_conv_layers(self, input_size):
                #the following code calculates and inserts input_size, and out_size attributes for each layer. 
                total_feature_maps = 1
                for layer in self.convNet_layers:
                        layer["input_size"] = input_size
                        if layer["layer_type"] == "conv":

                                total_feature_maps *= len(layer["filters"])
                                filter_shape = np.shape(layer["filters"][0])
                                padding = int(np.ceil(0.5*(filter_shape[0] -1)))
                                out_size = tuple([int(((input_size[i] + 2*padding - filter_shape[i])/layer["stride"]) + 1)for i in range(2)])
                                layer["bias"] = [np.random.rand(out_size[0], out_size[1]) for i in range(len(layer["filters"]))]
                                
                        else:
                                kernal_size = layer["ker_size"]
                                stride = layer["stride"]
                                padding = int(np.ceil(0.5*(kernal_size -1)))
                                out_size = tuple([int(((input_size[i] + 2*padding- kernal_size)/stride) + 1 )for i in range(2)])
                                
                        layer["out_size"] = out_size
                        input_size = out_size
                
                flat_net_outsize = np.prod(self.convNet_layers[-1]["out_size"])*total_feature_maps

                # self.change_input_size(new_input_size=flat_net_outsize)

                print("--> Convolutional layers initialised")
                return flat_net_outsize


