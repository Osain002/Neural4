import numpy as np


class tools:
    def map_array_to_positions(pos_matrix, arr):
        non_zero_pos = np.nonzero(pos_matrix)
        arr_vals = arr.flatten()
        diff = len(arr_vals) - len(non_zero_pos[0])
        for i, val in enumerate(arr_vals[:-diff]):
            pos_x, pos_y = non_zero_pos[0][i], non_zero_pos[1][i]
            pos_matrix[pos_x, pos_y] = val
        return pos_matrix

    def convolution(mat1, mat2, padding):

        m1_shape = np.shape(mat1)
        m2_shape = np.shape(mat2)

        if padding is None:
            padding = [int(np.ceil(0.5*(shape - 1))) for shape in m2_shape]

        mat1 = np.pad(mat1, padding[0])
        out_size = tuple(
            [int(((m1_shape[i] + 2*padding[i] - m2_shape[i] + 1)))for i in range(2)])
        out_mat = np.zeros(out_size)
        for i in range(len(mat1) - m2_shape[0]):
            for j in range(len(mat1[i]) - m2_shape[1]):
                kernal = mat1[i:i+m2_shape[0], j:j+m2_shape[1]]
                c = np.sum(np.multiply(kernal, mat2))
                out_mat[i][j] = c
        return out_mat


class Backprop_conv:

    def __init__(self, conv_layers, dense_layers):
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

    def get_deltas_from_dense(self):
        next_layer_delta = self.dense_layers[0]["deltas"]
        next_layer_weights = self.dense_layers[0]["weights"]
        w_d = np.matmul(next_layer_weights, next_layer_delta)
        df = self.dense_layers[0]["activation"][1]
        a = np.array(self.dense_layers[0]["input"])
        a = a/len(a)
        df = df(a)
        delta = np.multiply(df, w_d)
        shape_conv_deltas = np.shape(self.conv_layers[-1]["layer_out"])
        deltas = np.reshape(delta, shape_conv_deltas)
        self.conv_layers[-1]["deltas"] = deltas

    def compute_conv_deltas(self):
        deltas = self.conv_layers[-1]["deltas"]
        for i, layer in reversed(list(enumerate(self.conv_layers))):

            layer_type = layer["layer_type"]
            if layer_type == "pooling":  # ONLY IMPLENTED FOR MAX POOLING SO FAR...
                de_pool = layer["de_pool"]
                de_pooled_deltas = [tools.map_array_to_positions(
                    de_pool, delta) for delta in deltas]
                if i == len(self.conv_layers) - 1 or self.conv_layers[i+1] or self.conv_layers[i+1]["layer_type"] != "conv":
                    self.conv_layers[i-1]["deltas"] = de_pooled_deltas
                else:
                    self.conv_layers[i]["deltas"] = de_pooled_deltas
                    self.conv_layers[i]["filters"] = self.conv_layers[i+1]["filters"]
            if layer_type == "conv":
                if layer["deltas"] is None:
                    next_layer_deltas = self.conv_layers[i+1]["deltas"]
                    next_layer_weights = self.conv_layers[i+1]["filters"]
                    shape_deltas = np.shape(next_layer_deltas)
                    num_filters = len(next_layer_weights)
                    next_layer_deltas = np.reshape(next_layer_deltas, (int(
                        shape_deltas[0]/num_filters), num_filters, shape_deltas[1], shape_deltas[2]))
                    new_deltas = []
                    da = []
                    for r in layer["layer_out"]:
                        da.append([layer["activation_derivative"](val)
                                  for val in r])
                    for d in next_layer_deltas:
                        deltas = np.sum([tools.convolution(d[i], filter, None)
                                        for i, filter in enumerate(next_layer_weights)], 0)
                        new_deltas.append(deltas)
                    new_deltas = np.multiply(new_deltas, da)
                    layer["deltas"] = new_deltas
                deltas = layer["deltas"]

    def calc_partial_derivatives_conv(self):
        for i, layer in reversed(list(enumerate(self.conv_layers))):

            if layer["layer_type"] == "conv":
                prev_layer_out = self.conv_layers[i]["input"]
                deltas = self.conv_layers[i]["deltas"]

                shape_deltas = np.shape(deltas)
                num_filters = len(layer["filters"])

                dw = []

                if len(deltas) > num_filters:
                    deltas = np.reshape(deltas, (num_filters, int(
                        shape_deltas[0]/num_filters), shape_deltas[1], shape_deltas[2]))
                    for j, deltas_ in enumerate(deltas):
                        print(len(deltas), len(prev_layer_out))
                        try:
                            dw.append(np.sum(
                                [tools.convolution(d, prev_layer_out[j], padding=[1, 1]) for d in deltas_], 0))
                        except:
                            break
                else:
                    for d in deltas:
                        dw.append(tools.convolution(
                            d, prev_layer_out[0], [1, 1]))

                layer["dw"] = dw

    def update_conv_params(self, learning_rate):
        for i, layer in enumerate(self.conv_layers):
            if layer["layer_type"] == "conv":
                dw = layer["dw"]
                deltas = layer["deltas"]

                layer["filters"] = [np.subtract(f, dw[j])
                                    for j, f in enumerate(layer["filters"])]
