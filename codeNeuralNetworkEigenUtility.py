import eigen_face_loader
trainingData, testData = eigen_face_loader.load_data()
import neuralnetwork
net = neuralnetwork.Network([len(trainingData), 200, 15])
net.SGD(trainingData, 50, 5, .09, 15, test_data = testData)