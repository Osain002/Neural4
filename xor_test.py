from Neural_frame import Neural
import numpy as np


input_ = [[1,0], [0,0], [0,1], [1,1]]
actual_out_ = [[1], [0], [1], [0]]


Neural = Neural()

Neural.insert_dense(input_size=2, shape_layers=(3,8), activation= "tanh")
Neural.build_dense_layers(output_size=1, out_activation="sigmoid")
Neural.network_summary()

#train the net
Neural.train_dense_layers(x_train=input_, y_train=actual_out_, epochs=50, learning_rate= 0.1)

Neural.feed_forward_dense(input=[1,1])

net_out = Neural.get_network_out()

print("Net out:",net_out)
print("Target out:", [0])



Neural.plot_errors(save_location="Images/error_plot.png", show=True)