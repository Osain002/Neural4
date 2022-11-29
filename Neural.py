import Layers.Activations as act
import Layers.net_layers as net_layers
import Layer_operations.dense.forward_feed as dense_ff
import Layer_operations.dense.backprop as dense_bp
import Layer_operations.conv.forward_conv as conv_ff

import numpy as np
import matplotlib.pyplot as plt

Layers = net_layers.Layers
dense_ff = dense_ff.feed_forward
dense_bp = dense_bp.Backprop
conv_ff = conv_ff.ConvForward

import warnings

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore')


class Net_frame(Layers, dense_ff, dense_bp, conv_ff):

        def __init__(self) -> None:
                Layers.__init__(self)

                dense_ff.__init__(self, self.dense_layers)
                dense_bp.__init__(self, self.dense_layers)

                conv_ff.__init__(self, self.convNet_layers)

                self.errors = []

        def get_network_out(self):
                return self.dense_layers[-1]["layer_out"]
        
        def calculate_error(self, network_outs, targets):
                total_error = 0
                for i in range(len(targets)):
                        total_error += (network_outs[i] - targets[i])**2    
                self.errors.append(total_error)
                return total_error


        def plot_errors(self, save_location, show) -> None:
                x = np.arange(0, len(self.errors), step=1)
                fig, ax = plt.subplots()
                ax.plot(x[::10], self.errors[::10])
                ax.grid()
                fig.savefig(save_location)
                print("error_plot.png saved successfully")

                if show:
                        plt.show()
                
        
        def train_dense_layers(self, x_train, y_train,epochs, learning_rate):
                epoch = 0
                
                while epoch < epochs:
                        # if epoch > 2:
                        #         learning_rate = 1/(0.1*epoch)
                        for i, input in enumerate(x_train):
                                target = y_train[i]
                                self.feed_forward_dense(input)
                                net_out = self.dense_layers[-1]["layer_out"]
                                self.run_backprop(target, learning_rate)
                                self.calculate_error(net_out, target)
                        epoch += 1 
        

        def train_conv_network(self, x_train, y_train, epochs, learning_rate):

                epoch = 0
                while epoch < epochs: 
                        for i,input in enumerate(x_train):
                                target = y_train[i]
                                self.run_conv_forward([input])
                                conv_out = self.get_conv_out()
                                flat_data = self.flatten(conv_out)
                                flat_data = np.reshape(flat_data, len(flat_data))

                                self.feed_forward_dense(flat_data)
                                
                                
                                
                        epoch += 1

                                
                
                



