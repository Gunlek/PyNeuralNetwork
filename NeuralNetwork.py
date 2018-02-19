import numpy as np
import math as m


class NeuralNetwork:

    def __init__(self, input_number, hidden_number, output_number):
        self.input_number = input_number
        self.hidden_number = hidden_number
        self.output_number = output_number

        self.w_ih = np.random.rand(hidden_number, input_number)
        self.w_ho = np.random.rand(output_number, hidden_number)

        self.bias_o = np.random.rand(output_number, 1)
        self.bias_h = np.random.rand(hidden_number, 1)

        self.learning_rate = 0.1

    def predict(self, inputs):
        v_activation = np.vectorize(self.activation)
        hiddens = self.w_ih.dot(inputs)
        hiddens += self.bias_h
        hiddens = v_activation(hiddens)

        outputs = self.w_ho.dot(hiddens)
        outputs += self.bias_o
        outputs = v_activation(outputs)

        return outputs

    def activation(self, x):
        return 1 / (1 + m.exp(-x))

    def der_activation(self, y):
        return (y * (1 - y))

    def train(self, features, labels):
        v_activation = np.vectorize(self.activation)
        v_der_activation = np.vectorize(self.der_activation)

        hiddens = np.dot(self.w_ih, features)
        hiddens += self.bias_h
        hiddens = v_activation(hiddens)

        outputs = np.dot(self.w_ho, hiddens)
        outputs += self.bias_o
        outputs = v_activation(outputs)

        # Calculate errors
        outputs_errors = labels - outputs

        # Calculate gradient
        gradient = v_der_activation(outputs)
        gradient = np.dot(gradient, outputs_errors)
        gradient = np.dot(gradient, self.learning_rate)

        hidden_tranpose = hiddens.transpose()
        delta_ho = np.dot(gradient, hidden_tranpose)

        self.w_ho += delta_ho
        self.bias_o += gradient

        who_t = self.w_ho.transpose()
        hidden_errors = np.dot(who_t, outputs_errors)

        hidden_gradient = v_der_activation(hiddens).transpose()
        hidden_gradient = np.dot(hidden_gradient, hidden_errors)
        hidden_gradient = np.dot(hidden_gradient, self.learning_rate)

        inputs_transpose = labels.transpose()
        deltas_ih = np.dot(hidden_gradient, inputs_transpose)

        print(deltas_ih)

        self.w_ih += deltas_ih
        self.bias_h += hidden_gradient
