from Neural import Net_frame

input_ = [[1, 0], [0, 0], [0, 1], [1, 1]]
actual_out_ = [[1], [0], [1], [0]]

Neural = Net_frame()

# construct the dense layers
Neural.insert_dense(input_size=2, shape_layers=(1, 28), activation="relu")
Neural.build_dense_layers(output_size=1, out_activation="sigmoid")

Neural.network_summary()

# train the network
Neural.train_dense_layers(
    x_train=input_, y_train=actual_out_, epochs=100, learning_rate=0.2)

# test the network
Neural.feed_forward_dense(input=[1, 1])
net_out = Neural.get_network_out()


print("Net out:", net_out)
print("Target out:", [0])

# plot errors
Neural.plot_errors(save_location="Images/error_plot.png", show=True)
