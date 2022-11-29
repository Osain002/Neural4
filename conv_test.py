



from Neural import Net_frame
import numpy as np
from keras.datasets import mnist

import pprint

(train_X, train_y), (test_X, test_y) = mnist.load_data()


# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))




Neural = Net_frame()

#add convolution layers
Neural.add_convolution_layer(num_filters=2, filter_size=3, activation="relu")
Neural.add_convolution_layer(num_filters=2, filter_size=3, activation="relu")
Neural.add_pooling_layer(type="max", kernal_size=3, stride=3)
Neural.add_pooling_layer(type="max", kernal_size=3, stride=3)


# add dense layers
Neural.insert_dense(input_size=2, shape_layers=(1,128), activation= "relu")


#build network
Neural.build_layers(img_size=(28,28), out_size=1, out_activation="sigmoid")

#return network architecture
Neural.network_summary()

#train the network 
Neural.train_conv_network(x_train=train_X[:100], y_train=train_y[:100], epochs=1, learning_rate=1)


#get output of network
out = Neural.get_network_out()



print(out)



