import NeuralNetwork as nn
import numpy as np


network = nn.NeuralNetwork(2, 2, 1)

inputs = np.zeros((2, 1))
outputs = np.zeros((1, 1))

inputs[0, 0] = 1
inputs[1, 0] = 0
outputs[0, 0] = 1
network.train(inputs, outputs)

inputs[0, 0] = 0
inputs[1, 0] = 1
outputs[0, 0] = 1
network.train(inputs, outputs)

inputs[0, 0] = 0
inputs[1, 0] = 0
outputs[0, 0] = 0
network.train(inputs, outputs)

inputs[0, 0] = 1
inputs[1, 0] = 1
outputs[0, 0] = 0
network.train(inputs, outputs)

predict_inputs = np.zeros((2, 1))
predict_inputs[0, 0] = 0
predict_inputs[1, 0] = 0
print("prediction 0, 0: "+str(network.predict(predict_inputs)))

predict_inputs = np.zeros((2, 1))
predict_inputs[0, 0] = 0
predict_inputs[1, 0] = 1
print("prediction 0, 1: "+str(network.predict(predict_inputs)))

predict_inputs = np.zeros((2, 1))
predict_inputs[0, 0] = 1
predict_inputs[1, 0] = 1
print("prediction 1, 1: "+str(network.predict(predict_inputs)))
