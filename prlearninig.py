#! /usr/bin/python
import numpy as np
import copy
import time
import random
import math

f = open('results', 'w')
f.close()

class NeighboursSet:
	def __init__(self, inArray, inQ):
		self.content = inArray
		self.size = len(inArray)
		self.q = inQ

	def add(sample):
		self.content.append(sample)
		self.size += 1

	def remove(sample):
		self.content.remove(sample)
		self.size -= 1
 
	def classAssignment(self, point):
		self.distanceArray = []
		self.createDistanceArray(point)
		return self.determineClass()
	
	def createDistanceArray(self, point):
		for eachSample in self.content:
			self.distanceArray.append([self.distance(eachSample[0], point[0]), eachSample[1]])

	def determineClass(self):
		self.distanceArray.sort()
		tmp = self.distanceArray[0:self.q]
		determiner = []
		for item in tmp:
			determiner.append(item[1])
		return max(set(determiner), key=determiner.count)

	def distance(self, sample, point):
		return ((point-sample)*((point-sample).T)).item((0,0))

	def condenseSet(self):
		RS = copy.deepcopy(self.content)
		self.CS = []
		self.CS.append(RS[0])
		RS.remove(RS[0])
		for item in RS:
			dst = []
			for sample in self.CS:
				dst.append([self.distance(sample[0], item[0]), sample[1]])
			dst.sort()
			if dst[0][1] != item[1]:
				self.CS.append(item)
		self.content = copy.deepcopy(self.CS)

class Cluster:
	def __init__(self):
		self.content = []
		self.indices = [0 for i in range(10)]
		self.power = 0

	def add(self, matrix, label):
		matrix = matrix*1.0
		self.content.append([matrix, label])
		self.indices[label] += 1
		self.power += 1
		return 0

	def clusterLabel(self):
		label = self.indices.index(max(self.indices))
		number = self.indices[label]
		return label, number
	
	def compactation(self):
		label, number = self.clusterLabel()
		try:
			compactationRatio = float(number)/self.power
		except:
			return 0, 0
		return label, compactationRatio 
				
class WorkingSet:
	def __init__(self, inSet, inNumberOfSets, inNumberOfFeatures, inP, inQ, flag = 0):
		self.numberOfSets = inNumberOfSets
		self.numberOfFeatures = inNumberOfFeatures
		self.clusters = []
		for i in range(self.numberOfSets):
			self.clusters.append(Cluster())
		self.q = inQ
		self.representatns = int(math.ceil(float(inP)/10))
		self.path = inSet
		try:
			self.inputData = np.load(inSet)
		except IOError:
			print 'Could not find the given file. Make sure the path is correct.'
		self.instanceSize = len(self.inputData)

	def prepareNeighboursSet(self, P, q):
		self.dataSet = [[] for i in range (self.numberOfSets)]
		for i in self.inputData:
			self.dataSet[int(i[-1])].append(np.matrix(i[:-1]))
		reference = []
		for i in range (10):
			ran = [k for k in range(len(self.dataSet[i]))]
			random.shuffle(ran)
			ran = ran[:P]
			for j in ran:
				reference.append([self.dataSet[i][j], i])
		self.neighboursSet = NeighboursSet(reference, q)
		
	def qNN(self):
		for i, eachSet in enumerate(self.dataSet):
			print i
			for eachSample in eachSet:
				self.clusters[self.neighboursSet.classAssignment(eachSample)].add(eachSample, i)
		self.QNNResults = []
		for i in range (10):
			self.QNNResults.append(self.clusters[i].compactation())
		f = open('results', 'a')
		f.write( str(self.QNNResults)+'\n' )
		f.close()

	def prepeareKFold(self):
		dataS = []
		for eachData in self.inputData:
			dataS.append([ np.matrix(eachData[:-1]), int(eachData[-1]) ])	
		self.folds = [dataS[x:x+6000] for x in xrange(0, self.instanceSize, 6000)]

	def perceptron(self, stopping):
		self.prepeareKFold()
		for i in range(10):
			print 'in'
			f = open('results', 'a')
			f.write('Leave-one-out: '+str(i)+'\n')
			f.close
			self.constructSet(i)
			self.selectStopping(stopping)
			#self.convertToMatrices()
			self.trainPerceptron()
			res = self.evaluatePerceptron(self.out)
			f = open('results', 'a')
			f.write('\nExternal error ratio: '+str(res*100)+'%\n')
			f.close()
			print 'done'

	def constructSet(self, fold):
		self.out = self.folds[fold]
		self.trainingSet = []
		for i in range (10):
			if i != fold:
				#print self.trainingSet, len(self.folds[i])
				self.trainingSet += self.folds[i]

	def selectStopping(self, stopping):
		items = int(54000*stopping)
		generator = [i for i in range (54000)]
		random.shuffle(generator)
		generator = generator[:items]
		generator.sort()
		generator.reverse()
		self.stoppingTestSet = []
		for i in generator:
			self.stoppingTestSet.append(self.trainingSet[i])
			self.trainingSet.pop(i)

	def convertToMatrices(self):
		self.trainingSet = [[np.matrix(self.trainingSet[i][:-1]), int(self.trainingSet[i][-1])] for i in range(len(self.trainingSet))]
		self.stoppingTestSet = [[np.matrix(self.stoppingTestSet[i][:-1]), int(self.stoppingTestSet[i][-1])] for i in range(len(self.stoppingTestSet))]

	def trainPerceptron(self):
		self.mi = 0.1
		self.w = []
		for i in range(10):
			emptyMatrix = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			self.w.append(emptyMatrix)
		res = 0
		for rep in range(1000): ############################################################### <===========================|| 
			for eachSample in self.trainingSet:
				tmp = []
				for i in range(10):
					tmp.append((self.w[i]*eachSample[0].T).item((0,0)))
				mIndex = tmp.index(max(tmp))
				if mIndex != eachSample[1]:
					higher = []
					correctValue = tmp[eachSample[1]]
					for i, value in enumerate(tmp):
						if value > correctValue:
							higher.append(i)
					for h in higher:
						self.w[h] -= self.mi*eachSample[0]
					self.w[eachSample[1]] += self.mi*eachSample[0]
			newRes = self.evaluatePerceptron(self.stoppingTestSet)
			f = open('results', 'a')
			f.write('internel error ratio: '+str(newRes*100)+'%\n')
			f.close()
			if math.fabs(newRes - res) < 0.001:
				f = open('results', 'a')
				f.write('convergence achieved\n')
				f.close()
				break
		
			res = newRes
		return 0

	def evaluatePerceptron(self, evaluationSet):
		total = 0
		for eachSample in evaluationSet:
			tmp = []
			for i in range(10):
				tmp.append((self.w[i]*eachSample[0].T).item((0,0)))
			mIndex = tmp.index(max(tmp))
			if mIndex == eachSample[1]:
				total += 1
		res = 1-(float(total)/len(evaluationSet))
		#print len(evaluationSet)
		#print 'Ratio: ', res
		return res

	def constructEuclideanClassifier(self):
		self.numberOfTrials = []
		for i in self.dataSet:
			self.numberOfTrials.append(len(i))
		self.massCenters = []
		#for indexS, eachSubset in enumerate(self.dataSet):
		#	self.tmpAverage = (self.numberOfFeatures)*[0]
		#	for sample,eachSample in enumerate(eachSubset):
		#		for indexF, eachFeture in enumerate(eachSample):
		#			self.tmpAverage[indexF] += eachFeture
		for i, eachSubset in enumerate(self.dataSet):
			self.tmpAverage = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
			for eachSample in eachSubset:
				self.tmpAverage += eachSample
			self.tmpAverage /= float(self.numberOfTrials[i])
			self.massCenters.append(self.tmpAverage)
		return 0

	def constructStatisticalClassifier(self):
		self.aprioris = []
		for i in self.numberOfTrials:
			self.aprioris.append((float(i)/60000))
		self.lnapriori = []
		for apriori in self.aprioris:
			self.lnapriori.append(np.log(apriori))
		self.variances = []
		j = 0
		for eachClass in self.dataSet:
			print np.matrix(eachClass[0]), np.matrix(self.massCenters[j]).T, np.matrix(eachClass[0])
			print np.matrix(self.massCenters[j])
			tmpVariance = (np.matrix(eachClass[0]) - np.matrix(self.massCenters[j])).T * (np.matrix(eachClass[0]) - np.matrix(self.massCenters[j]))
			for i in range(1, self.numberOfTrials[j]):
				tmpVariance += (np.matrix(eachClass[i]) - np.matrix(self.massCenters[j])).T * (np.matrix(eachClass[i]) - np.matrix(self.massCenters[j]))
			tmpVariance /= float(self.numberOfTrials[j])
			j = j+1
			self.variances.append(tmpVariance)
		self.invMatrix = []
		for eachVariance in self.variances:
			self.invMatrix.append(np.linalg.inv(eachVariance))
		self.norms = []
		for eachVariance in self.variances:
			self.norms.append(np.linalg.norm(eachVariance))
		self.lnnorms = []
		for eachNorm in self.norms:
			self.lnnorms.append(np.log(eachNorm))
		return 0

	def gaussian(self, i, j):
		classTesting = []
		for k in range(0, self.numberOfSets):
			classTesting.append(( -0.5*self.lnnorms[k] -0.5*(np.matrix(self.dataSet[i][j]) - np.matrix(self.massCenters[k]))*self.invMatrix[k]*((np.matrix(self.dataSet[i][j]) - np.matrix(self.massCenters[k])).T) + self.lnapriori[k] ).tolist()[0][0])
		return (classTesting.index(max(classTesting)))

	def euclidean(self, i, j):
		classTesting = []
		for k in range(0, self.numberOfSets):

			dst = ((self.dataSet[i][j]-self.massCenters[k])*(self.dataSet[i][j]-self.massCenters[k]).T) 
			#classTesting.append(self.linearFunction(self.massCenters[k], self.dataSet[i][j]))
			classTesting.append(dst.item((0,0)))
		return (classTesting.index(min(classTesting)))

	def linearFunction(self,a,b):
		result = self.dotProduct(a,b) - 0.5*self.dotProduct(a,a)
		return result

	def combinedClassifier(self):
		self.clustersQNN = []
		for i in range(self.numberOfSets):
			self.clustersQNN.append(Cluster())
		self.clustersEuclidean = []
		for i in range(self.numberOfSets):
			self.clustersEuclidean.append(Cluster())
		self.clustersGaussian = []
		for i in range(self.numberOfSets):
			self.clustersGaussian.append(Cluster())

		for i, eachSet in enumerate(self.dataSet):
			print i
			for j, eachSample in enumerate(eachSet):
				fD = []
				fD.append(self.neighboursSet.classAssignment(eachSample))
				self.clustersQNN[fD[0]].add(eachSample, i)
				fD.append(self.euclidean(i,j))
				self.clustersEuclidean[fD[1]].add(eachSample, i)
				fD.append(self.gaussian(i,j))
				self.clustersGaussian[fD[2]].add(eachSample, i)
				self.clusters[max(set(fD), key = fD.count)].add(eachSample, i)
		self.QNNResults = []
		for i in range (10):
			self.QNNResults.append(self.clustersQNN[i].compactation())
		f = open('results', 'a')
		f.write('qNN classifier result:' + str(self.QNNResults)+'\n' )
		f.close()
		self.euclideanResults = []
		for i in range (10):
			self.euclideanResults.append(self.clustersEuclidean[i].compactation())
		f = open('results', 'a')
		f.write('Euclidean classifier result:' str(self.euclideanResults)+'\n' )
		f.close()
		self.gaussianResults = []
		for i in range (10):
			self.gaussianResults.append(self.clustersGaussian[i].compactation())
		f = open('results', 'a')
		f.write('Gaussian classifier result: ' str(self.gaussianResults)+'\n' )
		f.close()
		self.results = []
		for i in range (10):
			self.results.append(self.clusters[i].compactation())
		f = open('results', 'a')
		f.write('Combined classifier result:' str(self.results)+'\n' )
		f.close()

	def baggingInit(self, power, P, q):
		self.dataSet = [[] for i in range (self.numberOfSets)]
		for i in self.inputData:
			self.dataSet[int(i[-1])].append(np.matrix(i[:-1]))
		self.baggingSet = []
		for rep in range (power):
			reference = []
			for i in range (10):
				ran = [k for k in range(len(self.dataSet[i]))]
				random.shuffle(ran)
				ran = ran[:P]
				for j in ran:
					reference.append([self.dataSet[i][j], i])
			neighboursSet = NeighboursSet(reference, q)
			self.baggingSet.append(neighboursSet)

	def bagging(self, power, P, q):
		self.baggingInit(power, P, q)
		self.clustersBagging = []

		for rep in range (power+1):
			print rep
			clustersTmp = []
			for i in range(self.numberOfSets):
				clustersTmp.append(Cluster())
			self.clustersBagging.append(clustersTmp)
		print 'clusters done'
		for i, eachSet in enumerate(self.dataSet):
			print i
			for eachSample in eachSet:
				fD = []
				for rep in range (power):
					which = self.baggingSet[rep].classAssignment(eachSample)
					self.clustersBagging[rep][which].add(eachSample, i)
					fD.append(which)
				self.clustersBagging[power][max(set(fD), key=fD.count)].add(eachSample, i)

		for rep in range (power):
			self.Results = []
			for i in range (10):
				self.Results.append(self.clustersBagging[rep][i].compactation())
			f = open('results', 'a')
			f.write(str(rep)+ 'classifier result: ' +str(self.Results)+'\n' )
			f.close()
		self.Results = []
		for i in range (10):
			self.Results.append(self.clustersBagging[power][i].compactation())
		f = open('results', 'a')
		f.write('Combined classifier:' +str(self.Results)+'\n' )
		f.close()
		return 0

	def euclideanTest(self):
		self.clustersEuclidean = []
		for i in range(self.numberOfSets):
			self.clustersEuclidean.append(Cluster())
		for i, eachSet in enumerate(self.dataSet):
			print i
			for j, eachSample in enumerate(eachSet):
				fD = []
				fD.append(self.euclidean(i,j))
				self.clustersEuclidean[fD[0]].add(eachSample, i)
		self.euclideanResults = []
		for i in range (10):
			self.euclideanResults.append(self.clustersEuclidean[i].compactation())
		print self.euclideanResults
		




#inA = [[np.matrix([[0,0]]),0],[np.matrix([[10,10]]),1],[np.matrix([[10,0]]),2],[np.matrix([[0,10]]),3]]
#ns = NeighboursSet(inA, 1)
#print ns.classAssignment([np.matrix([[8,0]]),0])
print 'Patryk Karolak, Pattern Recognition\n'
print 'Attention. The significant output is logged to file called \'results\', the console output is just for the program status information'
action = ''
while (action != 'quit()' and action != 'Quit()' and action != 'QUIT()'): 
	print '\nTo perform certain action, type its number:\n'
	print '\t(1) Perceptron'
	print '\t(2) qNN'
	print '\t(3) Condensed qNN'
	print '\t(4) Bagging'
	print '\t(5) Combined Classifiers'
	print '\tquit() to close the program\n'
	action = raw_input('>>>')
	if action == '1' or action == '(1)':
		print '\nPerceptron\n'
		inFile = raw_input('Path to file *.npy (press ENTER to use default: datos_procesados.npy)\n>>> ')
		if (inFile == ''):
			inFile = 'datos_procesados.npy'
		inNumberOfSets = raw_input('Number of categories (press ENTER to use default: 10)\n>>> ')
		if (inNumberOfSets != ''):
			inNumberOfSets = int(inNumberOfSets)
		else:
			inNumberOfSets = 10
		inNumberOfFeatures = raw_input('Number of characteristic features (press ENTER to use default: 9)\n>>> ')
		if (inNumberOfFeatures != ''):
			inNumberOfFeatures = int(inNumberOfFeatures)
		else:
			inNumberOfFeatures = 9
		inP = raw_input('Part of samples for perceptron internal validation (0.5 by default)\n>>> ')
		if (inP != ''):
			inP = float(inP)
		else:
			inP = 0.5
		inQ = 10

		ws = WorkingSet(inFile, inNumberOfSets, inNumberOfFeatures, inP, inQ)
		ws.perceptron(inP)
		#ws.bagging(6, 40, 20)
		#ws.prepareNeighboursSet(10,10)
		#ws.constructEuclideanClassifier()
		#ws.constructStatisticalClassifier()
		#ws.euclideanTest()
		#ws.combinedClassifier()
		#ws.prepareNeighboursSet(100,30)
		#ws.combinedClassifier()
		#print 'condensing'
		#ws.neighboursSet.condenseSet()
		#print len(ws.neighboursSet.content)
		#ws.qNN()

		#print len(ws.neighboursSet.content)
		#ws.perceptron(0.9)
	if action == '2' or action == '(2)':
		print '\nqNN\n'
		inFile = raw_input('Path to file *.npy (press ENTER to use default: datos_procesados.npy)\n>>> ')
		if (inFile == ''):
			inFile = 'datos_procesados.npy'
		inNumberOfSets = raw_input('Number of categories (press ENTER to use default: 10)\n>>> ')
		if (inNumberOfSets != ''):
			inNumberOfSets = int(inNumberOfSets)
		else:
			inNumberOfSets = 10
		inNumberOfFeatures = raw_input('Number of characteristic features (press ENTER to use default: 9)\n>>> ')
		if (inNumberOfFeatures != ''):
			inNumberOfFeatures = int(inNumberOfFeatures)
		else:
			inNumberOfFeatures = 9
		inP = raw_input('Number of each class representants for qNN (10 by default)\n>>> ')
		if (inP != ''):
			inP = int(inP)
		else:
			inP = 10
		inQ = raw_input('Number of neighbours (10 by default)\n>>> ')
		if (inQ != ''):
			inQ = int(inQ)
		else:
			inQ = 10

		ws = WorkingSet(inFile, inNumberOfSets, inNumberOfFeatures, inP, inQ)
		#ws.bagging(6, 40, 20)
		ws.prepareNeighboursSet(inP,inQ)
		#ws.constructEuclideanClassifier()
		#ws.constructStatisticalClassifier()
		#ws.euclideanTest()
		#ws.combinedClassifier()
		#ws.prepareNeighboursSet(100,30)
		#ws.combinedClassifier()
		#print 'condensing'
		#ws.neighboursSet.condenseSet()
		#print len(ws.neighboursSet.content)
		ws.qNN()

		#print len(ws.neighboursSet.content)
		#ws.perceptron(0.5)
		#ws.perceptron(0.9)
	if action == '3' or action == '(3)':
		print '\nCondensed qNN\n'
		inFile = raw_input('Path to file *.npy (press ENTER to use default: datos_procesados.npy)\n>>> ')
		if (inFile == ''):
			inFile = 'datos_procesados.npy'
		inNumberOfSets = raw_input('Number of categories (press ENTER to use default: 10)\n>>> ')
		if (inNumberOfSets != ''):
			inNumberOfSets = int(inNumberOfSets)
		else:
			inNumberOfSets = 10
		inNumberOfFeatures = raw_input('Number of characteristic features (press ENTER to use default: 9)\n>>> ')
		if (inNumberOfFeatures != ''):
			inNumberOfFeatures = int(inNumberOfFeatures)
		else:
			inNumberOfFeatures = 9
		inP = raw_input('Number of each class representants for qNN (10 by default)\n>>> ')
		if (inP != ''):
			inP = int(inP)
		else:
			inP = 10
		inQ = raw_input('Number of neighbours (10 by default)\n>>> ')
		if (inQ != ''):
			inQ = int(inQ)
		else:
			inQ = 10

		ws = WorkingSet(inFile, inNumberOfSets, inNumberOfFeatures, inP, inQ)
		#ws.bagging(6, 40, 20)
		ws.prepareNeighboursSet(inP,inQ)
		#ws.constructEuclideanClassifier()
		#ws.constructStatisticalClassifier()
		#ws.euclideanTest()
		#ws.combinedClassifier()
		#ws.prepareNeighboursSet(100,30)
		#ws.combinedClassifier()
		#print 'condensing'
		ws.neighboursSet.condenseSet()
		#print len(ws.neighboursSet.content)
		ws.qNN()

		#print len(ws.neighboursSet.content)
		#ws.perceptron(0.5)
		#ws.perceptron(0.9)
	if action == '4' or action == '(4)':
		print '\nBagging\n'
		inFile = raw_input('Path to file *.npy (press ENTER to use default: datos_procesados.npy)\n>>> ')
		if (inFile == ''):
			inFile = 'datos_procesados.npy'
		inNumberOfSets = raw_input('Number of categories (press ENTER to use default: 10)\n>>> ')
		if (inNumberOfSets != ''):
			inNumberOfSets = int(inNumberOfSets)
		else:
			inNumberOfSets = 10
		inNumberOfFeatures = raw_input('Number of characteristic features (press ENTER to use default: 9)\n>>> ')
		if (inNumberOfFeatures != ''):
			inNumberOfFeatures = int(inNumberOfFeatures)
		else:
			inNumberOfFeatures = 9
		inP = raw_input('Number of each class representants for qNN bagging (10 by default)\n>>> ')
		if (inP != ''):
			inP = int(inP)
		else:
			inP = 10
		inQ = raw_input('Number of neighbours (10 by default)\n>>> ')
		if (inQ != ''):
			inQ = int(inQ)
		else:
			inQ = 10
		clas = raw_input('Number of independent cassifiers (5 by default)\n>>> ')
		if (clas != ''):
			clas = int(clas)
		else:
			clas = 5

		ws = WorkingSet(inFile, inNumberOfSets, inNumberOfFeatures, inP, inQ)
		ws.bagging(clas, inP, inQ)
		#ws.prepareNeighboursSet(10,10)
		#ws.constructEuclideanClassifier()
		#ws.constructStatisticalClassifier()
		#ws.euclideanTest()
		#ws.combinedClassifier()
		#ws.prepareNeighboursSet(100,30)
		#ws.combinedClassifier()
		#print 'condensing'
		#ws.neighboursSet.condenseSet()
		#print len(ws.neighboursSet.content)
		#ws.qNN()

		#print len(ws.neighboursSet.content)
		#ws.perceptron(0.5)
		#ws.perceptron(0.9)
	if action == '5' or action == '(5)':
		print '\nCombined Classifiers\n'
		inFile = raw_input('Path to file *.npy (press ENTER to use default: datos_procesados.npy)\n>>> ')
		if (inFile == ''):
			inFile = 'datos_procesados.npy'
		inNumberOfSets = raw_input('Number of categories (press ENTER to use default: 10)\n>>> ')
		if (inNumberOfSets != ''):
			inNumberOfSets = int(inNumberOfSets)
		else:
			inNumberOfSets = 10
		inNumberOfFeatures = raw_input('Number of characteristic features (press ENTER to use default: 9)\n>>> ')
		if (inNumberOfFeatures != ''):
			inNumberOfFeatures = int(inNumberOfFeatures)
		else:
			inNumberOfFeatures = 9
		inP = raw_input('Number of each class representants for qNN bagging (10 by default)\n>>> ')
		if (inP != ''):
			inP = int(inP)
		else:
			inP = 10
		inQ = raw_input('Number of neighbours (10 by default)\n>>> ')
		if (inQ != ''):
			inQ = int(inQ)
		else:
			inQ = 10

		ws = WorkingSet(inFile, inNumberOfSets, inNumberOfFeatures, inP, inQ)
		#ws.bagging(clas, inP, inQ)
		ws.prepareNeighboursSet(inP,inQ)
		ws.constructEuclideanClassifier()
		ws.constructStatisticalClassifier()
		#ws.euclideanTest()
		ws.combinedClassifier()
		#ws.prepareNeighboursSet(100,30)
		#ws.combinedClassifier()
		#print 'condensing'
		#ws.neighboursSet.condenseSet()
		#print len(ws.neighboursSet.content)
		#ws.qNN()

		#print len(ws.neighboursSet.content)
		#ws.perceptron(0.5)
		#ws.perceptron(0.9)
