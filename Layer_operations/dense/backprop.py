import numpy as np



class Backprop:

        
        def calc_output_deltas(self, layers, target_out):
                net_out = layers[-1]["layer_out"]
                out_dx = layers[-1]["activation"][1]
                N = len(net_out)
                a = layers[-1]["a"]
                y_diff = np.subtract(net_out,target_out)/N
                da = out_dx(np.array(a))
                delta = np.multiply(y_diff, da)
                layers[-1]["deltas"] = delta



        def calc_layer_deltas(self, layers):
                for i, layer in reversed(list(enumerate(layers[:-1]))):
                        next_layer_delta = layers[i + 1]["deltas"]
                        next_layer_weights = layers[i+1]["weights"]
                        w_d = np.matmul(next_layer_weights, next_layer_delta)
                        df = layer["activation"][1]
                        a = layer["a"]
                        df = df(a)
                        delta = np.multiply(df, w_d)
                        layer["deltas"] = delta

                        

        def compute_partial_derivatives(self, layers):
                for i, layer in reversed(list(enumerate(layers))):
                        if i > 0:

                                prev_layer_out = layers[i-1]["layer_out"]                                
                        if i == 0:
                                prev_layer_out = layer["input"]
                        
                        delta = layer["deltas"]
                        dw = np.outer(delta, prev_layer_out).T
                        layer["dw"] = dw


        def update_params(self, layers, learning_rate):
                for i, layer in reversed(list(enumerate(layers, 1))):
                        weights = layer["weights"]
                        bias = layer["bias"]
                        delta = layer["deltas"]
                        dw = layer["dw"]

                        # print(np.shape(weights), np.shape(dw))

                        layer["weights"] = np.subtract(weights, learning_rate*dw)
                        layer["bias"] = np.subtract(bias, learning_rate*delta)

                        

                      

        
        def run_backprop(self, layers, net_out, target, learning_rate):
                
                
                # self.calculate_deltas(net_out=net_out, target_out=target, layers=layers)
                self.calc_output_deltas(layers, target)
                self.calc_layer_deltas(layers)
                self.compute_partial_derivatives(layers)

                self.update_params(layers, learning_rate)
        

