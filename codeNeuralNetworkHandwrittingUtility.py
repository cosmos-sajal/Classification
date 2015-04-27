'''
import mnist_loader
trainingData, validation_data, testData = mnist_loader.load_data_wrapper()
import copyCode
net = copyCode.Network([len(trainingData), 100, 10])
net.SGD(trainingData, 100, 30, .1, test_data = testData)
'''

import mnist_loader
trainingData, test_data = mnist_loader.load_data_wrapper()
import neuralnetwork
net = neuralnetwork.Network([784, 30, 10])
net.SGD(trainingData, 10, 10, 3.0, 10, test_data = test_data)