
from .dense_layers import dense_layers
from .convolutional import ConvLayers
import numpy as np



class Layers(dense_layers, ConvLayers):
        def __init__(self) -> None:
                dense_layers.__init__(self)
                ConvLayers.__init__(self)
        


        def build_network(self, conv_input_size, output_size, output_activation):
                self.build_dense_layers(output_size=output_size, out_activation=output_activation)
                if conv_input_size:
                        c_out_size = self.build_conv_layers(conv_input_size)
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
                        print("Input size:", dense_layers[0]["in_size"])
                        for layer in dense_layers:
                                print("Neurons:", layer["num_neurons"])
                                print("Activation:", layer["activation"])
                                print("============================")
                        print("")
                        print("")



# net = Layers()

# # net = dense_layers()

# # net.add_convolution_layer(num_filters=2, filter_size=3, activation='relu')
# # net.add_convolution_layer(num_filters=2, filter_size=3, activation='relu')
# # net.add_convolution_layer(num_filters=2, filter_size=3, activation='relu')
# # net.add_pooling_layer(type="max", kernal_size=3, stride=3)

# # net.build_conv_layers((28,28))
# net.insert_dense(10, (4,4), "relu")
# # net.build_dense_layers(output_size=3, out_activation="relu")

# net.build_network(None, 4, "relu")




# net.network_summary()