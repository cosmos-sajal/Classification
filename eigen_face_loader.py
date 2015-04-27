# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:59:21 2015

@author: pramod
"""
import numpy as numpy
import random
import glob
from scipy.misc import *

def LoadImages(directory, split):
    # get a list of all the picture filenames
    gifs = glob.glob(directory + '/*.gif')
    # uncomment the below line when trying an unknown file
    #extraGif = glob.glob("/media/cosmos/Data/College Notes/M.Tech/Semester 4/Statistical Methods in AI/Project - Face Recognition/2.gif")
    classMap = {}
    testGIF = []
    allLabels = []
    testLabels = []
    trainingGIF = []
    trainingLabels = []
    for i in range(len(gifs)):
        if random.random() < split:
            trainingGIF.append(gifs[i])
            l = gifs[i].split("/");
            labelName = l[len(l)-1].split(".")[0][-2:]
            trainingLabels.append(labelName)
            allLabels.append(labelName)
        else:
            testGIF.append(gifs[i])
            l = gifs[i].split("/");
            labelName = l[len(l)-1].split(".")[0][-2:]
            testLabels.append(labelName)
            allLabels.append(labelName)
    # uncomment the below 2 lines when trying an unknown file
    #testGIF.append(extraGif[0])
    #testLabels.append("un")
    #allLabels.append("un")
    trainingImgs = numpy.array([imread(i, True).flatten() for i in trainingGIF])
    testImgs = numpy.array([imread(i, True).flatten() for i in testGIF])
    # creating a list of class labels
    allLabels = set(allLabels)
    noOfClasses = len(allLabels)
    sortedLabels = []
    for i in allLabels:
        sortedLabels.append(i)
    sortedLabels = sorted(sortedLabels)
    # creating a mapping for confusion matrix
    j = 0
    for i in sortedLabels:
        classMap[i] = j
        j = j + 1
    return trainingImgs,testImgs,trainingLabels,testLabels,noOfClasses,classMap
    # mean
    mu = numpy.mean(data, 0)
    # mean adjust the data
    ma_data = data - mu
    # run SVD
    e_faces, sigma, v = numpy.linalg.svd(ma_data.transpose(), full_matrices=False)
    # compute weights for each image
    weights = numpy.dot(ma_data, e_faces)
    return e_faces, weights, mu
def PCA(data):
    # mean
    mu = numpy.mean(data, 0)
    # mean adjust the data
    ma_data = data - mu
    # run SVD
    e_faces, sigma, v = numpy.linalg.svd(ma_data.transpose(), full_matrices=False)
    # compute weights for each image
    weights = numpy.dot(ma_data, e_faces)
    return e_faces, weights, mu
def InputWeight(testData, mu, e_faces):
    ma_data = testData - mu
    weights = numpy.dot(ma_data, e_faces)
    return weights
def load_data():
    inDIR  = "/home/cosmos/CSStuff/SMAI/Project 2 - Classification/input/yalefaces"
    outDIR = "/home/cosmos/CSStuff/SMAI/Project 2 - Classification/output"
    imgDims = (243, 320)
    
    split = 0.8
    trainingData, testData, trainingLabels, testLabels, noOfClasses, classMap = LoadImages(inDIR, split)
    e_faces, trainingWeights, mu = PCA(trainingData)
    #print trainingWeights.shape[0]
    #print trainingWeights.shape
    #print mu.shape
    #print e_faces.shape
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    print tr_d[0].shape
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
    '''
    testWeights=numpy.zeros((len(testData),trainingWeights.shape[1]))
    #print testWeights.shape
    #print trainingWeights.shape
    #print len(testWeights)
    for i in range(len(testWeights)):
        #testWeight =InputWeight(testData[i],mu,e_faces)
        testWeights[i]=InputWeight(testData[i],mu,e_faces)
    
    #formatting weights
        training_inputs = [numpy.reshape(x, (len(trainingWeights), 1)) for x in trainingWeights]
        test_inputs= [numpy.reshape(x,(len(trainingWeights) , 1)) for x in testWeights]
    #Convert training labels to vectors
    trainingLabels=numpy.asarray(trainingLabels)
    trainingLabels=trainingLabels.astype(numpy.float)
    testLabels=numpy.asarray(testLabels)
    testLabels=testLabels.astype(numpy.float)
    #decrementing subject labels
    for i in range(len(trainingLabels)):
        trainingLabels[i]=trainingLabels[i]-1
    for i in range(len(testLabels)):
        testLabels[i]=testLabels[i]-1
    #print testLabels
    #print vectorized_result(trainingLabels[0])
    #print trainingWeights.shape
    training_results = [vectorized_result(y) for y in trainingLabels]
    #test_results= [vectorized_result(y) for y in testLabels]
    
    tr_d=zip(training_inputs, training_results)
    te_d=zip(test_inputs,testLabels)
    #print te_d
    return (tr_d,te_d)
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = numpy.zeros((15, 1))
    e[j-1] = 1.0
    return e
#load_data()
    

    

