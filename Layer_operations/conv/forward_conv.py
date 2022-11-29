
import numpy as np

class ConvForward:
        def __init__(self, conv_layers):
                self.conv_layers = conv_layers
        
        def pad(self, px_matrix, shape_weights):
                num_padding = int(np.ceil(0.5*(shape_weights -1)))
                px_matrix = np.pad(px_matrix, num_padding, mode='constant')
                return px_matrix
        
        def convolve(self, input, layer):
                filters = layer["filters"]
                kernal_i, kernal_j = np.shape(filters[0])
                input = self.pad(input, kernal_i)
                layer_outs = []
                for k,filter in enumerate(filters):
                        out = np.zeros(layer["out_size"])
                        for i in range(len(input) - kernal_i + 1):
                                for j in range(len(input) - kernal_j + 1):
                                        kernal = input[i:i + kernal_i, j:j + kernal_j]
                                        activation = layer["activation"]
                                        conv = np.sum(np.multiply(kernal, filter))
                                        out[i][j] = activation([conv + layer["bias"][k][i][j]])[0]
                        layer_outs.append(out)
                return layer_outs

        
        def pooling(self, input, layer):
                ker_size = layer["ker_size"]
                stride = layer["stride"]
                pool_func = layer["pool_func"]
                out_size = layer["out_size"]
                pool = np.zeros(out_size)
                input = self.pad(input, ker_size)
                for i in range(0, len(input) - ker_size + 1, stride):
                        for j in range(0, len(input[i]) - ker_size + 1, stride):
                                kernal = input[i:i + ker_size, j:j + ker_size]
                                p = pool_func(kernal)
                                pool[int(i/stride)][int(j/stride)] = p
                return pool
        
                                
        
        def flatten(self, arr):
                return arr.flatten()
        
        def get_conv_out(self):
                return self.conv_layers[-1]["layer_out"]

        def run_conv_forward(self, input):

                for i,layer in enumerate(self.conv_layers):
                        
                        if layer["layer_type"] == "conv":
                                layer_out = np.array([self.convolve(fm, layer) for fm in input])
                                shape_0, shape_1, shape_2, shape_3 = np.shape(layer_out)
                                layer_out = np.reshape(layer_out, (shape_0*shape_1, shape_2, shape_3))
                        elif layer["layer_type"] == "pooling":
                                layer_out = np.array([self.pooling(fm, layer) for fm in input])
                        layer["layer_out"] = layer_out
                        
                        input = layer_out
                        
                        
                        


                                



                

        