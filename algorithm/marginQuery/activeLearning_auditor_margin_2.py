"""
active learning for auditor with random initialization and min margin
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

modelName = "al_auditor_type6_random_2"
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp
# random.seed(3)


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

		self.ex_id = dd(list)

	def select_example(self, unlabeled_list):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)

		return random.sample(unlabeled_list, 1)[0]
 
		# print("unlabeledIdNum\t", unlabeledIdNum)
		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			# print("unlabeledId\t", unlabeledId)
			labelPredictProb = self.clf.predict_proba(self.fn[unlabeledId].reshape(1, -1))[0]
			# print("labelPredictProb", labelPredictProb)

			# labelIndexMap = {} ##labelIndex: labelProb
			# labelNum = len(labelPredictProb)
			# for labelIndex in range(labelNum):
			# 	labelIndexMap.setdefault(labelIndex, labelPredictProb[labelIndex])

			# sortedLabelIndexList = sorted(labelIndexMap, key=labelIndexMap.__getitem__, reverse=True)
			# maxLabelIndex = sortedLabelIndexList[0]
			# subMaxLabelIndex = sortedLabelIndexList[1]

			# coefDiff = 0
			# if labelNum == 2:
			# 	coefDiff = np.dot(self.clf.coef_, self.fn[unlabeledId])
			# 	idScore = np.abs(sigmoid(coefDiff)-1+sigmoid(coefDiff))

			# 	# print("idScore", idScore)
			# 	# print("diff", np.abs(labelPredictProb[0]-labelPredictProb[1]))
			# else:
			# 	maxCoef = self.clf.coef_[maxLabelIndex]
			# 	subMaxCoef = self.clf.coef_[subMaxLabelIndex]
			# 	# coefDiff = np.dot(maxCoef, self.fn[unlabeledId])-np.dot(subMaxCoef, self.fn[unlabeledId])
			# 	# idScore = (sigmoid(np.dot(maxCoef, self.fn[unlabeledId]))-sigmoid(np.dot(subMaxCoef, self.fn[unlabeledId])))/np.sum(sigmoid(np.dot(self.clf.coef_, self.fn[unlabeledId])))
			# 	idScore = (sigmoid(np.dot(maxCoef, self.fn[unlabeledId]))-sigmoid(np.dot(subMaxCoef, self.fn[unlabeledId])))
			# idScore = 1-idScore

				# print("idScore", idScore)

				# probList = sigmoid(np.dot(self.clf.coef_, self.fn[unlabeledId]))/np.sum(sigmoid(np.dot(self.clf.coef_, self.fn[unlabeledId])))
				# print("probList", probList)


			# print(labelPredictProb)
			sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)
			# print(sortedLabelPredictProb)
			maxLabelPredictProb = sortedLabelPredictProb[0]
			subMaxLabelPredictProb = sortedLabelPredictProb[1]
			# print("diff", maxLabelPredictProb-subMaxLabelPredictProb)
			# print("maxLabelPredictProb\t", maxLabelPredictProb)
			idScore = 1-(maxLabelPredictProb-subMaxLabelPredictProb)
			# idScore = coefDiff
			# print("idScore\t", idScore)

			unlabeledIdScoreMap[unlabeledId] = idScore
		# print("=======")
		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)
		print("selecting ", sortedUnlabeledIdList[0], self.label[sortedUnlabeledIdList[0]])
		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, fn_test, label_test, labeled_list):

		fn_train = self.fn[labeled_list]
		label_train = self.label[labeled_list]
		
		self.clf.fit(fn_train, label_train)
		fn_preds = self.clf.predict(fn_test)

		print(len(label_train), sum(label_train), "ratio", sum(label_train)*1.0/len(label_train))
		print("label_train", label_train)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

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

			self.clf = LR(random_state=3, fit_intercept=False)

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

			while queryIter < rounds:
				fn_train_iter = []
				label_train_iter = []

				fn_train_iter = self.fn[labeledExList]
				label_train_iter = self.label[labeledExList]

				self.clf.fit(fn_train_iter, label_train_iter) 

				idx = self.select_example(unlabeledExList) 
				print(queryIter, "idx", idx, self.label[idx])
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


if __name__ == "__main__":

	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('./selectedNameFeature4Label.txt').readlines()]
	# tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	# label = tmp[:,-1]
	# selectedInstanceIDList = [float(i.strip()) for i in open('selectedID4Label.txt').readlines()]
	label = np.array([float(i.strip()) for i in open('selectedTargetAuditorLabel.txt').readlines()])
	print("number of types", len(set(label)))
	print 'class count of true labels of all ex:\n', ct(label)

	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, fn, label)

	al.run_CV()

