
from .dense_layers import dense_layers
from .convolutional import ConvLayers
import numpy as np



class Layers(dense_layers, ConvLayers):
        def __init__(self) -> None:
                dense_layers.__init__(self)
                ConvLayers.__init__(self)
        


        def build_layers(self, **kwargs):
                
                img_size = kwargs["img_size"]
                output_size = kwargs["out_size"]
                out_activation = kwargs["out_activation"]

                if output_size:
                        self.build_dense_layers(output_size=output_size, out_activation=out_activation)
                
                if img_size:
                        c_out_size = self.build_conv_layers(img_size)
                        self.update_input_size(c_out_size)
                        
        
        def network_summary(self):
                conv_layers = self.convNet_layers
                dense_layers = self.dense_layers

                if conv_layers != []:
                        print("============================")
                        print("CONVOLUTION LAYERS")
                        print("============================")
                        for layer in conv_layers:
                                print("Layer type:", layer["layer_type"])

                                if layer["layer_type"] == "conv":
                                        print("Filter size:", layer["filter_size"])
                                elif layer["layer_type"] == "pooling":
                                        print("Kernal size:", layer["ker_size"])
                                
                                print("Input size: ", layer["input_size"])
                                print("Out size: ", layer["out_size"])
                                print("============================")
                                
                if dense_layers != []:
                        print("DENSELY CONNECTED LAYERS")
                        print("============================")
                        print("Network input size:", dense_layers[0]["in_size"])
                        print("============================")
                        for layer in dense_layers:
                                
                                print("Num neurons:", layer["num_neurons"])
                                print("Activation:", layer["activation"])
                                print("============================")
                        print("")
                        print("")


