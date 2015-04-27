"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm

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

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    confusionMatrix = CreateConfusionMatrix(predictions, test_data[1])
    avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, 10, len(test_data[1]))
    PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity)
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    

if __name__ == "__main__":
    svm_baseline()