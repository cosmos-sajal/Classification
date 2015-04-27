# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:40:07 2015

@author: pramod
"""
import numpy as np
import random as random
def CreateConfusionMatrix(predictions, testSet, noOfClasses):
    '''
    s=set()
    for i in testSet:
          s.add(i)
    l=list(s)
    '''
    confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
   
    '''
    d={}
    for x in range(noOfClasses):
        d[x] = x
    '''

    for x in range(len(testSet)):
        if predictions[x] == testSet[x]:
            confusionMatrix[testSet[x]][testSet[x]] = confusionMatrix[testSet[x]][testSet[x]] + 1 
        else:
            confusionMatrix[testSet[x]][predictions[x]] = confusionMatrix[testSet[x]][predictions[x]] + 1
    return confusionMatrix

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, noOfTestSamples):
    totalRecall = 0.0
    totalPrecision = 0.0
    totalSpecificity = 0.0
    recall = []
    precision = []
    specificity = []
    #print "Precisions for Different Classes"
    #print "________________________________"
    for i in range(0, len(confusionMatrix[0])):
        classPrecision = 0.0
        for j in range(0, len(confusionMatrix)):
            classPrecision = classPrecision + confusionMatrix[j][i]
        if classPrecision != 0.0:
            classPrecision = (confusionMatrix[i][i] / float(classPrecision)) * 100
        else:
            classPrecision = 0.0
        precision.append(classPrecision)
        #print "Class Precision for class", i + 1,":", classPrecision
        totalPrecision = totalPrecision + classPrecision
    #print "Recalls for Different Classes"
    #print "________________________________"
    for i in range(0, len(confusionMatrix)):
        classRecall = 0.0
        for j in range(0, len(confusionMatrix[i])):
            classRecall = classRecall + confusionMatrix[i][j]
        if classRecall != 0.0:
            classRecall = (confusionMatrix[i][i] / float(classRecall)) * 100
        else:
            classRecall = 0.0
        recall.append(classRecall)
        #print "Class Recall for class", i + 1,":", classRecall
        totalRecall = totalRecall + classRecall
    for i in range(0, len(confusionMatrix[0])):
        numerator = noOfTestSamples - confusionMatrix[i][i]
        denominator = numerator
        for j in range(0, len(confusionMatrix)):
            if i != j:
                denominator = denominator + confusionMatrix[j][i]
        classSpecificity = (numerator / float(denominator)) * 100
        totalSpecificity = totalSpecificity + classSpecificity
        specificity.append(classSpecificity)

    avgRecall = (totalRecall / float(noOfClasses))
    avgPrecision = (totalPrecision / float(noOfClasses))
    avgSpecificity = (totalSpecificity / float(noOfClasses))
    return avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity

def PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity):
    print "Confusion Matrix:"
    for i in range(0, len(confusionMatrix)):
        print confusionMatrix[i]
    for i in range(0, len(precision)):
        print "Class", i + 1
        print "--------"
        print "Precision :", precision[i]
        print "Recall :", recall[i]
        print "Specificity :", specificity[i]
        print "\n"
    print "-------------------"
    print "Average Recall:", avgRecall
    print "Average Precision:", avgPrecision
    print "Average Specificity:", avgSpecificity

def GetAccuracy(testLabels, predictions):
    correct = 0

    for x in range(len(testLabels)):
        if testLabels[x] == predictions[x]:
                  correct += 1
                                         
    return (correct/float(len(testLabels))) * 100.0 

class Network():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes #number of neurons in resepctive layers
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        '''
        print self.biases.shape
        print self.weights.shape
        
        self.biases = [random.random() for y in sizes[1:]]
        self.biases=np.asarray(self.biases)
        self.weights = [random.random() 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights=np.asarray(self.weights)
        '''
    #does a'=sigmoid(wa+b)
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a
    #Stochastic Gradient Descent
    '''
    training_data is list of tuples(x,y) representing training inputs and corresponding desired outputs 
    epochs is number of epochs you traain data for
    size of minibatch - size of mini batch ised when sampling 
    test_data - will evaluate network after each training
    '''

    def SGD(self, training_data, epochs, mini_batch_size, eta, noOfClasses, 
                test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                sum,test_results=self.evaluate(test_data)
                print "Epoch {0}: {1} / {2}".format(
                    j, sum, n_test)
            else:
                print "Epoch {0} complete".format(j)
        #print test_results
        predictions=[int(i[0]) for i in test_results]
        testLabels=[int(i[1]) for i in test_results]
        confusionMatrix = CreateConfusionMatrix(testLabels,predictions, noOfClasses)
        avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, len(testLabels))
        PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity)
        accuracy = GetAccuracy(predictions, testLabels)
        print "Accuracy :", accuracy

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        #print test_results
        return sum(int(x == y) for (x, y) in test_results),test_results
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)