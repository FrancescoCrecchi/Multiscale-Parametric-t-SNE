import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense as fc


def kl_divergence(P, Y, n_dims=2, batch_size=100):
    alpha = n_dims - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(1e-14)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def simple_dnn(n_input, n_output):
    model = Sequential()
    model.add(fc(500, input_dim=n_input, activation='relu'))
    model.add(fc(500, activation='relu'))
    model.add(fc(2000, activation='relu'))
    model.add(fc(n_output))
    model.compile(loss=kl_divergence, optimizer="adam")
    return model
