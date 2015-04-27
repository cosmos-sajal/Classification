# -*- coding: utf-8 -*-

import csv
import random
import math
import operator 
#import statistics
import numpy as np
def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)
def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5
def prepareMatrix(confMat,l):
    confMat_row=[]
    #print('length' +repr(l))
    for row in range (l):
        for col in range(l):
            confMat_row.append(0)
        confMat.append(confMat_row)
        confMat_row=[]
    return confMat
def confusionMatrix(predictions,testSet,confMat):
    #print("testSet"+repr(testSet[-1]))
    
    s=set()
    for row in testSet:
          s.add(row[-1])
    l=list(s)
    #print('list is'+repr(l))
    d={}       
    for x in range(len(l)):
        d[l[x]]=x
    
   # if count==0:
    confMat=prepareMatrix(confMat,len(l))
        #print('Conf Matrix init')
    #for row in confMat:
        #print(row)    
    
    #for x in range(len(testSet)):
        #print ('Predicted '+predictions[x] + ' Actual '+testSet[x][-1])
    for x in range(len(testSet)):
        if predictions[x]==testSet[x][-1]:
            #print("hey"+repr(d[predictions[x]]))
            #print('index'+repr(d[testSet[x][-1]])+"&"+repr(d[testSet[x][-1]]))
            confMat[d[testSet[x][-1]]][d[testSet[x][-1]]]+=1
             
            
            
        else:
            confMat[d[predictions[x]]][d[testSet[x][-1]]]+=1
    #v=[1,2,3]
    
    
    return confMat

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        lenOfAttr=len(dataset[0])
        for x in range(len(dataset)-1):
            for y in range(lenOfAttr-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
         

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    
    for x in range(len(trainingSet)):
        #print(trainingSet[x])
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        #print('response'+response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0] 

def getAccuracy(testSet, predictions):
    correct = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
                  correct += 1
                                         
    return (correct/float(len(testSet))) * 100.0            
 
 
def main(): 
    filename='sonar-all.data'
    #s=set()
    #with open(filename, 'rt') as csvfile:
     #   lines = csv.reader(csvfile)
      #  dataset = list(lines)
       # for row in dataset:
        #    s.add(row[-1])
    '''l=list(s)
    print('list is'+repr(l))
    d={}
    for x in range(len(l)):
        d[l[x]]=x''' 
# prepare data
    accuracyList=[]
    confMat=[]  
    confMatAll=[]
    count=0
    for i in range(10):
        trainingSet=[]
        testSet=[]
          
        split = 0.5
        loadDataset(filename, split, trainingSet, testSet)
        print ("Train set: " + repr(len(trainingSet)));
        print ('Test set: ' + repr(len(testSet)))
        
        
    #for row in range(len(testSet)): 
     #   s.add(testSet[row][-1])
   
    
    # generate predictions
        predictions=[]
        k =1
        print('length of test set', len(testSet))
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
            #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
            
    #print(getAccuracy(testSet,predictions))
            
        confMat=confusionMatrix(predictions,testSet,confMat)
        count+=1;
        confMatAll.append(confMat)
        for row in confMat:
            print(row)
        confMat=[]
        accuracy = getAccuracy(testSet, predictions)
        accuracyList.append(accuracy)
        print('Accuracy: ' + repr(accuracy) + '%')
    
    #print(accuracyList)
    print('Mean Accuracy '+repr(sum(accuracyList)/float(len(accuracyList))))
    print('Standard Deviation '+repr(pstdev(accuracyList)))
    print('Confusion Matrix:')
    a=np.zeros(shape=(len(confMatAll[0]),len(confMatAll[0])))
    for row in confMatAll:
        #print(row)
        #print
        b=np.array(row)
        a+=b
    
    for row in a:
        print(row)
    
        

main()

