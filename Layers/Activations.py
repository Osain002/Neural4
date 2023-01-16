from scipy.special import softmax
import numpy as np


class Activations:

    def sigmoid(vec):
        sig_vec = []
        for x in vec:
            sig = 1/(1+np.exp(-x))
            sig_vec.append(sig)
        return sig_vec

    def sigmoid_derivative(vec):
        v = Activations.sigmoid(vec)

        sig_der = []
        for x in v:
            s = x*(1-x)
            sig_der.append(s)

        return sig_der

    def softplus(vec):
        for i, x in enumerate(vec):
            vec[i] = np.log(1 + np.exp(-x))
        return vec

    def softplus_derivative(vec):
        return Activations.sigmoid(vec)

    def relu(vec):
        relu_vec = []

        for x in vec:
            if x <= 0:
                relu_vec.append(0.01*x)
            else:
                relu_vec.append(x)
        return np.array(relu_vec)

    def relu_derivative(vec):
        rd = []
        for i in range(len(vec)):
            if vec[i] < 0:
                rd.append(0)
            else:
                rd.append(1)
        return rd

    def softmax(vec):
        return softmax(vec)

    def softmax_derivative(vec):
        d = Activations.softmax(vec)
        for i in range(len(d)):
            vec[i] = d[i] - d[i]**2
        return vec

    def tanh(vec):
        return np.tanh(vec)

    def tanh_derivative(vec):
        one = np.ones_like(vec)
        tanh_squared = np.multiply(
            Activations.tanh(vec), Activations.tanh(vec))

        return one - tanh_squared

    def id(vec):
        return vec

    def id_der(vec):
        return np.ones_like(vec)
