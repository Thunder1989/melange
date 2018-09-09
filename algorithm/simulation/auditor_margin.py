"""
active learning with random initialization and least confidence query strategy
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from datetime import datetime

modelName = "auditor_margin"
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp

def sigmoid(x):
  	  return (1 / (1 + np.exp(-x)))

def get_name_features(names):

		name = []
		for i in names:
			s = re.findall('(?i)[a-z]{2,}',i)
			name.append(' '.join(s))

		cv = CV(analyzer='char_wb', ngram_range=(3,4))
		fn = cv.fit_transform(name).toarray()

		return fn
class active_learning:

	def __init__(self, fold, rounds, fn, label):

		self.fold = fold
		self.rounds = rounds

		self.fn = fn
		self.label = label

		self.tao = 0
		self.alpha_ = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.ex_id = dd(list)

	def select_example(self, unlabeled_list):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)
		# print("---------------")
		alpha = 0.1
		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			labelPredictProb = self.m_clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)

			maxLabelPredictProb = sortedLabelPredictProb[0]
			subMaxLabelPredictProb = sortedLabelPredictProb[1]

			idScore = maxLabelPredictProb-subMaxLabelPredictProb
			idScore = -idScore
			# selectCB = self.get_confidence_bound(unlabeledId)
			# print("selectCB", self.m_selectCbRate*selectCB)
			# LCB = maxLabelPredictProb+self.m_selectCbRate*selectCB

			# idScore = np.dot(self.m_clf.coef_, self.fn[unlabeledId])-2*alpha*selectCB

			# idScore = -selectCB

			# print(np.dot(self.m_clf.coef_, self.fn[unlabeledId]), 2*selectCB, 2*alpha*selectCB)
			unlabeledIdScoreMap[unlabeledId] = idScore
		# exit()
		# sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)
		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.m_clf.fit(fn_train, label_train)
		fn_preds = self.m_clf.predict(fn_test)

		# print(len(label_train), sum(label_train), "ratio", sum(label_train)*1.0/len(label_train))
		# print("label_train", label_train)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def init_confidence_bound(self, featureDim):
		self.m_A = self.m_lambda*np.identity(featureDim)
		self.m_AInv = np.linalg.inv(self.m_A)

	def update_confidence_bound(self, exId):
		self.m_A += np.outer(self.fn[exId], self.fn[exId])
		self.m_AInv = np.linalg.inv(self.m_A)

	def get_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.fn[exId], self.m_AInv), self.fn[exId]))

		return CB

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		print("featureNum", len(self.fn[0]))
		print("non zero feature num", sum(self.fn[0]))

		totalTransferNumList = []
		np.random.seed(3)
		np.random.shuffle(indexList)

		foldNum = 10
		foldInstanceNum = int(totalInstanceNum*1.0/foldNum)
		foldInstanceList = []

		for foldIndex in range(foldNum-1):
			foldIndexInstanceList = indexList[foldIndex*foldInstanceNum:(foldIndex+1)*foldInstanceNum]
			foldInstanceList.append(foldIndexInstanceList)

		foldIndexInstanceList = indexList[foldInstanceNum*(foldNum-1):]
		foldInstanceList.append(foldIndexInstanceList)
		# kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		cvIter = 0
		# random.seed(3)
		totalAccList = [[] for i in range(10)]
		for foldIndex in range(foldNum):
			
			# self.clf = LinearSVC(random_state=3)

			self.m_clf = LR(random_state=3, fit_intercept=False)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)
			
			fn_test = self.fn[test]
			label_test = self.label[test]

			fn_train = self.fn[train]

			initExList = []
			random.seed(3)
			initExList = random.sample(train, 3)
			fn_init = self.fn[initExList]
			label_init = self.label[initExList]
			print("initExList\t", initExList, label_init)

			queryIter = 3
			labeledExList = []
			unlabeledExList = []
			###labeled index
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			featureDim = len(self.fn[0])
			self.init_confidence_bound(featureDim)

			while queryIter < rounds:
				fn_train_iter = []
				label_train_iter = []

				fn_train_iter = self.fn[labeledExList]
				label_train_iter = self.label[labeledExList]

				self.m_clf.fit(fn_train_iter, label_train_iter) 

				idx = self.select_example(unlabeledExList)
				self.update_confidence_bound(idx) 
				# print(queryIter, "idx", idx, self.label[idx])
				labeledExList.append(idx)
				unlabeledExList.remove(idx)

				acc = self.get_pred_acc(fn_test, label_test, labeledExList)
				totalAccList[cvIter].append(acc)
				queryIter += 1

			cvIter += 1      
		
		totalACCFile = modelVersion+".txt"
		f = open(totalACCFile, "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

def readFeatureLabel(featureLabelFile):
	f = open(featureLabelFile)

	featureMatrix = []
	labelList = []

	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)

		featureSample = []
		for lineIndex in range(lineLen-1):
			featureVal = float(line[lineIndex])
			featureSample.append(featureVal)

		labelList.append(int(line[lineLen-1]))

		featureMatrix.append(featureSample)

	f.close()

	return featureMatrix, labelList

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	transferLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		transferLabelList.append(int(line[1]))

	f.close()

	return transferLabelList

if __name__ == "__main__":

	featureDim = 20
	sampleNum = 500
	classifierNum = 5

	featureLabelFile = "./simulatedFeatureLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"
	# featureLabelFile = "./simulatedFeatureLabel.txt"
	featureMatrix, labelList = readFeatureLabel(featureLabelFile)
	featureMatrix = np.array(featureMatrix)
	labelArray = np.array(labelList)

	transferLabelFile = "./simulatedTransferLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"
	# transferLabelFile = "./simulatedTransferLabel.txt"
	transferLabelList = readTransferLabel(transferLabelFile)
	transferLabelArray = np.array(transferLabelList)
	# print(labelList)
	auditorLabel = (labelArray==transferLabelArray)*1.0
	# print(auditorLabel)
	# exit()
	# label = np.array([float(i.strip()) for i in open('targetAuditorLabel.txt').readlines()])

	# tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	# label = tmp[:,-1]
	print('class count of true labels of all ex:\n', ct(labelArray))
	print("count of auditor", ct(auditorLabel))
	# exit()
	# mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	# fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, featureMatrix, auditorLabel)

	al.run_CV()
