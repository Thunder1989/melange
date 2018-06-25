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

modelName = "activeLearning_random"
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

		self.fn = np.array(fn)
		self.label = np.array(label)

		self.tao = 0
		self.alpha_ = 1

		self.ex_id = dd(list)

		self.m_lambda = 0.01
		self.m_selectA = 0
		self.m_selectAInv = 0
		self.m_selectCbRate = 0.002 ###0.005
		self.clf = 0

	def select_example(self, unlabeled_list):
		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		return random.sample(unlabeled_list, 1)[0]

		# print("debug")
		# return sortedUnlabeledIdList[0]
		# unlabeledIDScoreMap2 = {}

		# print("---------------")
		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			# print("unlabeledId\t", unlabeledId)
			labelPredictProb = self.clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]

			# print(self.clf.coef_)

			# labelIndexMap = {} ##labelIndex: labelProb
			# labelNum = len(labelPredictProb)
			# for labelIndex in range(labelNum):
			# 	labelIndexMap.setdefault(labelIndex, labelPredictProb[labelIndex])

			# sortedLabelIndexList = sorted(labelIndexMap, key=labelIndexMap.__getitem__, reverse=True)
			# print("labelPredictProb\t", labelPredictProb)
			# sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)
			# # print(sortedLabelPredictProb)
			# maxLabelPredictProb = sortedLabelPredictProb[0]
			# subMaxLabelPredictProb = sortedLabelPredictProb[1]
			# maxLabelIndex = sortedLabelIndexList[0]
			# subMaxLabelIndex = sortedLabelIndexList[1]

			selectCB = self.get_select_confidence_bound(unlabeledId)
				# probDiff = sigmoid(np.dot(maxCoef, self.fn[unlabeledId])-self.m_selectCbRate*selectCB)-sigmoid(np.dot(subMaxCoef, self.fn[unlabeledId])+self.m_selectCbRate*selectCB)
			# print(coefDiff, selectCB, self.m_selectCbRate*selectCB)
			# LCB = coefDiff-2*0.002*selectCB

			# LCB = coefDiff-2*self.m_selectCbRate*selectCB
			idScore = selectCB
			# idScore = 1-probDiff
			# print("idScore", idScore)

			# print(maxLabelIndex, subMaxLabelIndex)
		
			# marginProb = maxLabelPredictProb-subMaxLabelPredictProb


			# print("selectCB", self.m_selectCbRate*selectCB)
			# LCB = maxLabelPredictProb+self.m_selectCbRate*selectCB
				
			# LCB = sigmoid(coefDiff) -2*selectCB

			# idScore = LCB
			# unlabeledIDScoreMap2[unlabeledId] = 1-coefDiff
			unlabeledIdScoreMap[unlabeledId] = idScore

		# sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)
		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		# sortedUnlabeledIdList2 = sorted(unlabeledIDScoreMap2, key=unlabeledIDScoreMap2.__getitem__, reverse=True)

		# for unlabeledIdIndex in range(unlabeledIdNum):
		# 	unlabeledId = sortedUnlabeledIdList[unlabeledIdIndex]
		# 	unlabeledId2 = sortedUnlabeledIdList2[unlabeledIdIndex]

		# 	print(sortedUnlabeledIdList[unlabeledIdIndex], unlabeledIdScoreMap[unlabeledId], sortedUnlabeledIdList2[unlabeledIdIndex], unlabeledIDScoreMap2[unlabeledId2])

		return sortedUnlabeledIdList[0]

	def init_confidence_bound(self, featureDim):
		self.m_selectA = self.m_lambda*np.identity(featureDim)
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def update_select_confidence_bound(self, exId):
		# print("updating select cb", exId)
		self.m_selectA += np.outer(self.fn[exId], self.fn[exId])
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def get_select_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.fn[exId], self.m_selectAInv), self.fn[exId]))

		return CB

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.clf.fit(fn_train, label_train)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

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
		totalAccList = [[] for i in range(10)]
		totalNewClassFlagList = [[] for i in range(10)]
		for foldIndex in range(foldNum):
			# self.clf = LinearSVC(random_state=3)

			self.clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3, fit_intercept=False)

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

			featureDim = len(fn_train[0])
			self.init_confidence_bound(featureDim)
			
			initExList = []
			random.seed(110)
			initExList = random.sample(train, 3)
			fn_init = self.fn[initExList]
			label_init = self.label[initExList]

			newClassFlagList = []
			existClassList = []

			for newLabel in label_init:
				if newLabel not in existClassList:
					existClassList.append(newLabel)
					newClassFlagList.append(1)
				else:
					newClassFlagList.append(0)

			print("initExList\t", initExList, label_init)
			queryIter = 3
			labeledExList = []
			unlabeledExList = []
			###labeled index
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			while queryIter < rounds:
				fn_train_iter = []
				label_train_iter = []

				fn_train_iter = self.fn[labeledExList]
				label_train_iter = self.label[labeledExList]

				self.clf.fit(fn_train_iter, label_train_iter) 

				idx = self.select_example(unlabeledExList) 
				self.update_select_confidence_bound(idx)
				# print(queryIter, "idx", idx, self.label[idx])
				# self.update_select_confidence_bound(idx)

				newLabel = self.label[idx]
				if newLabel not in existClassList:
					existClassList.append(newLabel)
					newClassFlagList.append(1)
				else:
					newClassFlagList.append(0)

				labeledExList.append(idx)
				unlabeledExList.remove(idx)

				acc = self.get_pred_acc(fn_test, label_test, labeledExList)
				totalAccList[cvIter].append(acc)
				queryIter += 1

			totalNewClassFlagList[cvIter] = newClassFlagList
			cvIter += 1      
			
		totalACCFile = modelVersion+"_acc.txt"
		f = open(totalACCFile, "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

		newClassFlagFile = modelVersion+"_newClassFlag.txt"
		f = open(newClassFlagFile, "w")
		for i in range(10):
			totalAlNum = len(totalNewClassFlagList[i])
			for j in range(totalAlNum):
				f.write(str(totalNewClassFlagList[i][j])+"\t")
			f.write("\n")
		f.close()

if __name__ == "__main__":

	f = open('./simulatedFeatureLabel.txt')
	featureMatrix = []
	label = []
	for rawLine in f:
		featureLine = rawLine.strip().split("\t")
		featureNum = len(featureLine)
		featureList = []
		for featureIndex in range(featureNum-1):
			featureVal = float(featureLine[featureIndex])
			featureList.append(featureVal)

		labelVal = float(featureLine[featureNum-1])

		featureMatrix.append(featureList)
		label.append(labelVal)

	f.close()

	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, featureMatrix, label)

	al.run_CV()
