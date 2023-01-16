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
        print(next_layer_delta)
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

    def de_pool_deltas(self, layer):
        de_pool_pos_matrix = layer["de_pool"]
        layer_deltas = layer["deltas"]
        new_deltas = []

        for delta in layer["deltas"]:

            de_pool_delta = tools.map_array_to_positions(
                de_pool_pos_matrix, delta)

            new_deltas.append(de_pool_delta)

    def compute_conv_deltas(self, layer):
        pass

    def compute_conv_derivatives(self, layer):
        pass

    def update_conv_params(self):
        pass

    def backprop_conv(self):
        self.get_deltas_from_dense()
        layers = self.conv_layers
        for i, layer in reversed(list(enumerate(layers))):
            if layer["layer_type"] == "pooling":
                self.de_pool_deltas(layer)
            elif layer["layer_type"] == "conv":
                self.compute_conv_deltas(layer)
            break
