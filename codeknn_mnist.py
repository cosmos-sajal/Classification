import mnist_loader
import os
import sys
import pdb
import glob
import math
import numpy
import random
import operator
from sets import Set
import scipy as scipy
from scipy.misc import *
from scipy import linalg

# Prints the confustion matrix
def getNeighbors(trainingSet, testInstance, trainingLabels, k):
    distances = []
    length = len(testInstance)-1
    
    for x in range(len(trainingSet)):
        #print(trainingSet[x])
        dist = EuclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist, trainingLabels[x]))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][2])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        #print('response'+response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def CreateConfusionMatrix(predictions, testSet):
    noOfClasses = 10
    confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
    s=set()
    for i in testSet:
          s.add(i)
    l=list(s)
    d={}
    for x in range(len(l)):
        d[l[x]]=x

    for x in range(len(testSet)):
        if predictions[x]==testSet[x]:
            confusionMatrix[d[testSet[x]]][d[testSet[x]]] = confusionMatrix[d[testSet[x]]][d[testSet[x]]] + 1 
        else:
            confusionMatrix[d[testSet[x]]][d[predictions[x]]] = confusionMatrix[d[testSet[x]]][d[predictions[x]]] + 1
    return confusionMatrix

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

def EuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def GetAccuracy(testLabels, predictions):
    correct = 0

    for x in range(len(testLabels)):
        if testLabels[x] == predictions[x]:
                  correct += 1
                                         
    return (correct/float(len(testLabels))) * 100.0    

def main(k):
	training_data, validation_data, test_data = mnist_loader.load_data()
	k = int(k[0])
	predictions=[]
	for i in range(	len(test_data[0])):
		neighbors=getNeighbors(training_data[0],test_data[0][i],training_data[1],k)
		result=getResponse(neighbors)
		predictions.append(result)
	confusionMatrix = CreateConfusionMatrix(predictions,test_data[1])
	avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, 10, len(test_data[1]))
	PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity)
	accuracy = GetAccuracy(test_data[1], predictions)
	print "Accuracy :", accuracy

main(sys.argv[1:])