import numpy as np



class Backprop:
        def __init__(self, dense_layers):
                self.layers = dense_layers

        def calc_output_layer_deltas(self, target_out):

                net_out = self.layers[-1]["layer_out"]
                out_dx = self.layers[-1]["activation"][1]
                N = len(net_out)
                a = self.layers[-1]["a"]
                y_diff = np.subtract(net_out,target_out)/N
                da = out_dx(np.array(a))
                delta = np.multiply(y_diff, da)
                self.layers[-1]["deltas"] = delta
                
                
                

        def calc_hidden_layer_deltas(self):
                
                for i, layer in reversed(list(enumerate(self.layers[:-1]))):
                        next_layer_delta = self.layers[i + 1]["deltas"]
                        next_layer_weights = self.layers[i+1]["weights"]

                        w_d = np.matmul(next_layer_weights, next_layer_delta)
                        
                        df = layer["activation"][1]
                        a = np.array(layer["a"])

                        a = a/len(a)

                        df = df(a)                 
                        delta = np.multiply(df, w_d)
                        
                        layer["deltas"] = delta
                        

        def compute_partial_derivatives(self):
                for i, layer in reversed(list(enumerate(self.layers))):
                        if i > 0:
                                prev_layer_out = self.layers[i-1]["layer_out"]                                
                        if i == 0:
                                prev_layer_out = layer["input"]
                        delta = layer["deltas"]
                        dw = np.outer(delta, prev_layer_out).T
                        layer["dw"] = dw

        def update_params(self, learning_rate):
                for i, layer in reversed(list(enumerate(self.layers, 1))):
                        weights = layer["weights"]
                        bias = layer["bias"]
                        delta = layer["deltas"]
                        dw = layer["dw"]

                        layer["weights"] = np.subtract(weights, learning_rate*dw)
                        layer["bias"] = np.subtract(bias, learning_rate*delta)

        
        def run_backprop(self, target, learning_rate):
                
                self.calc_output_layer_deltas(target)
                self.calc_hidden_layer_deltas()
                self.compute_partial_derivatives()
                self.update_params(learning_rate)
        

