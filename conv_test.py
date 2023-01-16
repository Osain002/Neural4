

from Neural import Net_frame
import numpy as np
from keras.datasets import mnist

import pprint

(train_X, train_y), (test_X, test_y) = mnist.load_data()


tr_y = []
for i, d in enumerate(train_y):
    new = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new[d] = 1
    tr_y.append(new)

train_y = tr_y


# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))


Neural = Net_frame()

# add convolution layers
Neural.add_convolution_layer(num_filters=10, filter_size=3, activation="relu")
Neural.add_pooling_layer(type="max", kernal_size=3, stride=3)


Neural.add_convolution_layer(num_filters=10, filter_size=3, activation="relu")
Neural.add_pooling_layer(type="max", kernal_size=3, stride=3)


# add dense layers
Neural.insert_dense(input_size=1600, shape_layers=(3, 28), activation="relu")


# build network
Neural.build_layers(img_size=(28, 28), out_size=10, out_activation="softmax")

# return network architecture
Neural.network_summary()

# train the network
Neural.train_conv_network(
    x_train=[train_X[1000]/255], y_train=[train_y[1000]], epochs=1, learning_rate=1)


# Neural.get_deltas_from_dense()
# Neural.compute_conv_deltas()
# Neural.calc_partial_derivatives_conv()
# Neural.update_conv_params(learning_rate=0.1)


# Neural.plot_errors(save_location="t.png", show=True)
# get output of network
out = Neural.get_network_out()


print(out)
