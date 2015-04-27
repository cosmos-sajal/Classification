import sys
import numpy
import random
from scipy.misc import *
from scipy import linalg
from subprocess import call

def PopulateClassMap(classes):
	classMap = {}
	for i, val in enumerate(classes):
		classMap[val] = i
	return classMap

def LoadData(argv):
	fileName = ''.join(argv)
	data = []
	with open(fileName) as f:
		for line in f:
			data.append(line.strip('\n').split(','))
	data = numpy.concatenate((numpy.array(data)[:,[1,2,3,4]], numpy.array(data)[:,[0]]), axis=1)
	numOfRecords = len(data)
	numOfFeatures = len(data[0])
	classes = list(set([list(i) for i in zip(*data)][numOfFeatures - 1]))
	numOfClasses = len(classes)
	classMap = PopulateClassMap(classes)
	return data, numOfRecords, numOfFeatures, classMap

def DivideData(data, numOfRecords, numOfFeatures, classMap):
	random.shuffle(data)
	testLabels = []
	trainingLabels = []
	trainingData = data[ : (4 * numOfRecords) / 5 ]
	testData = data[ ((4 * numOfRecords) / 5) : ]
	for i in range(0, len(trainingData)):
		trainingLabels.append(classMap[trainingData[i][numOfFeatures - 1]])
	for i in range(0, len(testData)):
		testLabels.append(classMap[testData[i][numOfFeatures - 1]])
	return trainingData, testData, trainingLabels, testLabels

def WriteIntoFile(data, numOfFeatures, classMap, directory, fileType):
	if fileType == "train":
		fd = open(directory + "/training.txt", 'w+')
	else:
		fd = open(directory + "/test.txt", 'w+') 
	for i in range(0, len(data)):
		line = classMap[data[i][numOfFeatures - 1]]
		for j in range(0, len(data[i]) - 1):
			line = str(line) + " " + str(j + 1) + ":" + str(data[i][j])
		fd.write(line + "\n")
	fd.close()

def TrainUsingSVM(directory):
	filePath = directory + "/training.txt"
	command = directory + "/svm-train"
	call([command, "-t", "0", filePath])

def PredictLabels(directory):
	command = directory + "output/svm-predict"
	testFilePath = directory + "output/test.txt"
	modelPath = directory + "/training.txt.model"
	fd = open(directory + "/output/result", 'w+')
	resultFilePath = directory + "output/result"
	call([command, testFilePath, modelPath, resultFilePath])
	fd.close()

def CalculateAccuracy(directory, actualLabels):
	noOfCorrectlyClassifiedSamples = 0
	fd = open(directory + "/result", "r")
	predictedLabels = []
	for line in fd:
		predictedLabels.append(line)
	for i in range(0, len(predictedLabels)):
		predictedLabels[i] = int(predictedLabels[i][0 : len(predictedLabels[i]) - 1])
	for i in range(0, len(actualLabels)):
		if actualLabels[i] == predictedLabels[i]:
			noOfCorrectlyClassifiedSamples = noOfCorrectlyClassifiedSamples + 1
	accuracy = (noOfCorrectlyClassifiedSamples / float(len(actualLabels))) * 100
	return actualLabels, predictedLabels, accuracy

def CalculateConfusionMatrix(actualLabels, predictedLabels, noOfClasses):
	confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
	for i in range(0, len(actualLabels)):
		confusionMatrix[actualLabels[i]][predictedLabels[i]] = confusionMatrix[actualLabels[i]][predictedLabels[i]] + 1
	print confusionMatrix
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

def PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity, accuracy, iterationNo):
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
	print "Accuracy for iteration number", iterationNo + 1,":", accuracy

def main(argv, iterationNo):
	directory = argv[0]
	inDIR = directory + "input/"
	outDIR = directory + "output"
	data, numOfRecords, numOfFeatures, classMap = LoadData(inDIR + "wine.data.txt")
	noOfClasses = len(classMap)
	trainingData, testData, trainingLabels, testLabels = DivideData(data, numOfRecords, numOfFeatures, classMap)
	WriteIntoFile(trainingData, numOfFeatures, classMap, outDIR, "train")
	TrainUsingSVM(outDIR)
	WriteIntoFile(testData, numOfFeatures, classMap, outDIR, "test")
	PredictLabels(directory)
	actualLabels, predictedLabels, accuracy = CalculateAccuracy(outDIR, testLabels)
	confusionMatrix = CalculateConfusionMatrix(actualLabels, predictedLabels, noOfClasses)
	avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, len(testData))
	PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity, accuracy, iterationNo)
	return accuracy

total = 0.0
for i in range(0, 5):
	accuracy = main(sys.argv[1:], i)
	total = total + accuracy
	print "________________________________________________"
print "Average Accuracy : ",total / 5.0,"%" 