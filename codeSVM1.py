import sys
import glob
import numpy
import random
from scipy.misc import *
from scipy import linalg
from subprocess import call

def SplitData(directory):
	split = 0.8
	gifs = glob.glob(directory + '/*.gif')
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
	trainingImgs = numpy.array([imread(i, True).flatten() for i in trainingGIF])
	testImgs = numpy.array([imread(i, True).flatten() for i in testGIF])
	allLabels = set(allLabels)
	noOfClasses = len(allLabels)
	sortedLabels = []
	for i in allLabels:
		sortedLabels.append(i)
	sortedLabels = sorted(sortedLabels)
    # creating a mapping for confusion matrix
	j = 0
	for i in sortedLabels:
		if i[0] == '0':
			i = i[1 : ]
		classMap[i] = j
		j = j + 1
	return trainingImgs, testImgs, trainingLabels, testLabels, noOfClasses, classMap

def PCA(data):
    # mean
	mu = numpy.mean(data, 0)
    # mean adjust the data
	ma_data = data - mu
    # run SVD
	e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)
    # compute weights for each image
	weights = numpy.dot(ma_data, e_faces)
	return e_faces, weights, mu

def WriteIntoFile(data, labels, directory, fileType):
	if fileType == "train":
		fd = open(directory + "/training.txt", 'w+')
	else:
		fd = open(directory + "/test.txt", 'w+') 
	for i in range(0, len(data)):
		line = labels[i]
		for j in range(0, len(data[i])):
			line = line + " " + str(j + 1) + ":" + str(data[i][j])
		fd.write(line + "\n")
	fd.close()

def TrainUsingSVM(directory):
	filePath = directory + "/training.txt"
	command = directory + "/svm-train"
	call([command, "-t", "0", filePath])

def InputWeight(testData, mu, e_faces):
    ma_data = testData - mu
    weights = numpy.dot(ma_data, e_faces)
    return weights

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
		predictedLabels[i] = predictedLabels[i][0 : len(predictedLabels[i]) - 1]
	for i in range(0, len(actualLabels)):
		if actualLabels[i][0] == '0':
			actualLabels[i] = actualLabels[i][1 : ]
	for i in range(0, len(actualLabels)):
		if actualLabels[i] == predictedLabels[i]:
			noOfCorrectlyClassifiedSamples = noOfCorrectlyClassifiedSamples + 1
	accuracy = (noOfCorrectlyClassifiedSamples / float(len(actualLabels))) * 100
	return actualLabels, predictedLabels, accuracy

def CalculateConfusionMatrix(actualLabels, predictedLabels, classMap, noOfClasses):
	confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
	for i in range(0, len(actualLabels)):
		confusionMatrix[classMap[actualLabels[i]]][classMap[predictedLabels[i]]] = confusionMatrix[classMap[actualLabels[i]]][classMap[predictedLabels[i]]] + 1
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
	inDIR = directory + "input/yalefaces"
	outDIR = directory + "output"
	trainingData, testData, trainingLabels, testLabels, noOfClasses, classMap = SplitData(inDIR)
	e_faces, trainingWeights, mu = PCA(trainingData)
	WriteIntoFile(trainingWeights, trainingLabels, outDIR, "train")
	TrainUsingSVM(outDIR)
	testWeights = InputWeight(testData, mu, e_faces)
	WriteIntoFile(testWeights, testLabels, outDIR, "test")
	PredictLabels(directory)
	actualLabels, predictedLabels, accuracy = CalculateAccuracy(outDIR, testLabels)
	confusionMatrix = CalculateConfusionMatrix(actualLabels, predictedLabels, classMap, noOfClasses)
	avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, len(testData))
	PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity, accuracy, iterationNo)
	return accuracy

total = 0.0
for i in range(0, 5):
	accuracy = main(sys.argv[1:], i)
	total = total + accuracy
	print "________________________________________________"
print "Average Accuracy : ",total / 5.0,"%" 