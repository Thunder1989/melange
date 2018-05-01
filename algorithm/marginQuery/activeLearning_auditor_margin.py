"""
proactive learning with random initialization, judge classifier to judge whether transfer learning is correct. 
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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from datetime import datetime

modelName = "active_auditor_margin"
timeStamp = datetime.now()
timeStamp = str(timeStamp.month)+str(timeStamp.day)+str(timeStamp.hour)+str(timeStamp.minute)

modelVersion = modelName+"_"+timeStamp
# random.seed(3)

def get_name_features(names):

		name = []
		for i in names:
			s = re.findall('(?i)[a-z]{2,}',i)
			name.append(' '.join(s))

		cv = CV(analyzer='char_wb', ngram_range=(3,4))
		fn = cv.fit_transform(name).toarray()

		return fn

def sigmoid(x):
  	  return (1 / (1 + np.exp(-x)))

class _ActiveLearning:

	def __init__(self, fold, rounds, sourceDataFeature, sourceLabel, targetDataFeature, targetLabel, targetNameFeature):

		self.m_fold = fold
		self.m_rounds = rounds

		self.m_sourceDataFeature = sourceDataFeature
		self.m_sourceLabel = sourceLabel

		self.m_targetDataFeature = targetDataFeature
		self.m_targetNameFeature = targetNameFeature
		self.m_targetLabel = targetLabel

		self.m_randomForest = 0
		self.m_tao = 0

		self.m_alpha = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.05 ##0.05

		self.m_judgeClassifier = 0
		self.m_clf = 0

	def select_example(self, unlabeled_list):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore
		unlabeledIdNum = len(unlabeled_list)
		printNum = 10
		printIndex = 0

		# targetLabelMap = {} ###targetLabel: instanceIDList

		# return random.sample(unlabeled_list, 1)[0]

		# for unlabeledIdIndex in range(unlabeledIdNum):
		# 	unlabeledId = unlabeled_list[unlabeledIdIndex]

		# 	targetLabel = self.m_targetLabel[unlabeledId]

		# 	if targetLabel not in targetLabelMap:
		# 		targetLabelMap.setdefault(targetLabel, [])

		# 	targetLabelMap[targetLabel].append(unlabeledId)

		# targetLabelNumMap = ct(self.m_targetLabel[unlabeled_list])
		# # sortedTargetLabelList = sorted(targetLabelNumMap, key=targetLabelNumMap.__getitem__, reverse=True)

		# sortedTargetLabelList = sorted(targetLabelNumMap, key=targetLabelNumMap.__getitem__, reverse=True)
		# # sortedTargetLabelList = sorted(targetLabelNumMap, key=targetLabelNumMap.__getitem__)

		# print(sortedTargetLabelList)
		# # print("targetLabelNumMap", targetLabelNumMap)
		
		# mostInstanceTargetLabel = sortedTargetLabelList[0]
		# print("mostInstanceTargetLabel", mostInstanceTargetLabel)
		# # print(debug)
		# instanceList4TargetLabel = targetLabelMap[mostInstanceTargetLabel]

		# for instanceID in instanceList4TargetLabel:
		# 	labelPredictProb = self.m_judgeClassifier.predict_proba(self.m_targetNameFeature[instanceID].reshape(1, -1))[0]
		# 	sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)

		# 	maxLabelPredictProb = sortedLabelPredictProb[0]
		# 	subMaxLabelPredictProb = sortedLabelPredictProb[1]

		# 	idScore = 1-(maxLabelPredictProb-subMaxLabelPredictProb)

		# 	unlabeledIdScoreMap[instanceID] = idScore

		# sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		# print("selecting", sortedUnlabeledIdList[0])
		# return sortedUnlabeledIdList[0]

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = unlabeled_list[unlabeledIdIndex]
			
			labelPredictProb = self.m_judgeClassifier.predict_proba(self.m_targetNameFeature[unlabeledId].reshape(1, -1))[0]
		
			sortedLabelPredictProb = sorted(labelPredictProb, reverse=True)
		
			maxLabelPredictProb = sortedLabelPredictProb[0]
			subMaxLabelPredictProb = sortedLabelPredictProb[1]
			# print("maxLabelPredictProb\t", maxLabelPredictProb)
			idScore = 1-(maxLabelPredictProb-subMaxLabelPredictProb)
			# print("idScore\t", idScore)
			unlabeledIdScoreMap[unlabeledId] = idScore

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def get_pred_acc(self, targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter):

		# targetNameFeatureTrain = self.m_targetNameFeature[labeledIdList]
		# targetLabelTrain = self.m_targetLabel[labeledIdList]
		self.m_judgeClassifier.fit(targetNameFeatureIter, targetLabelIter)
		# self.m_clf.fit(targetNameFeatureIter, targetLabelIter)
		# targetLabelPreds = self.m_clf.predict(targetNameFeatureTest)
		# print(targetNameFeatureIter, targetLabelIter)
		print(len(targetLabelIter), sum(targetLabelIter), "pos ratio", sum(targetLabelIter)*1.0/len(targetLabelIter))
		print(self.m_judgeClassifier.coef_)
		targetAuditorPreds = self.m_judgeClassifier.predict(targetNameFeatureTest)

		acc = accuracy_score(targetLabelTest, targetAuditorPreds)
		precision = precision_score(targetLabelTest, targetAuditorPreds)
		recall = recall_score(targetLabelTest, targetAuditorPreds)
		# print("acc\t", acc)
		# print debug
		return acc, precision, recall

	def get_base_learners(self):
		self.m_randomForest = RFC(n_estimators=100, criterion='entropy', random_state=3)

		self.m_randomForest.fit(self.m_sourceDataFeature, self.m_sourceLabel)

	def run_CV(self):

		cvIter = 0
		
		totalInstanceNum = len(self.m_targetLabel)
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
		# random.seed(3)
		# totalAccList = [[] for i in range(10)]
		# humanAccList = [[] for i in range(10)]

		self.get_base_learners()

		totalAuditorPrecisionList = [[] for i in range(10)]
		totalAuditorRecallList = [[] for i in range(10)]
		totalAuditorAccList = [[] for i in range(10)]

		targetNameFeature = self.m_targetNameFeature
		targetDataFeature = self.m_targetDataFeature
		targetTransferLabel = self.m_randomForest.predict(targetDataFeature)
		targetAuditorLabel = 1.0*(self.m_targetLabel == targetTransferLabel)

		selectedInstanceAuditorLabelMap = {}

		for instanceIndex in range(totalInstanceNum):
			auditorLabel = targetAuditorLabel[instanceIndex]
			targetLabel = self.m_targetLabel[instanceIndex]

			if targetLabel not in selectedInstanceAuditorLabelMap:
				selectedInstanceAuditorLabelMap.setdefault(targetLabel, [])

			selectedInstanceAuditorLabelMap[targetLabel].append(auditorLabel)

		for targetLabel in selectedInstanceAuditorLabelMap:
			print("selected target label", targetLabel, ct(selectedInstanceAuditorLabelMap[targetLabel]))

		exit()
		for foldIndex in range(foldNum):
			auditorMap = {} ##class: (neg, pos)

			self.m_judgeClassifier = LR(random_state=3)
			# self.m_judgeClassifier = LinearSVC(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)

			targetNameFeatureTrain = self.m_targetNameFeature[train]
			# targetAuditorLabelTrain = targetAuditorLabel[train]

			targetNameFeatureTest = self.m_targetNameFeature[test]
			targetAuditorLabelTest = targetAuditorLabel[test]

			posTargetAuditorNum = sum(targetAuditorLabelTest)
			posRatio = posTargetAuditorNum*1.0/len(targetAuditorLabelTest)
			print("posRatio\t", posRatio)

			sourceUniqueClass = np.unique(self.m_sourceLabel)

			initExList = []
			random.seed(5)
			initExList = random.sample(train, 3)
			print("initExList\t", initExList)

			targetNameFeatureInit = self.m_targetNameFeature[initExList]
			targetAuditorLabelInit = targetAuditorLabel[initExList]

			queryIter = 0
			labeledExList = []
			unlabeledExList = []
			###labeled index
			labeledExList.extend(initExList)
			unlabeledExList = list(set(train)-set(labeledExList))

			activeLabelNum = 3.0
			transferLabelNum = 0.0
			transferFeatureList = []
			transferFlagList = []

			featureDim = len(targetNameFeatureTrain[0])
			# self.init_confidence_bound(featureDim)

			targetNameFeatureIter = targetNameFeatureInit
			targetAuditorLabelIter = targetAuditorLabelInit

			auditorPrecisionList = []
			auditorRecallList = []
			auditorAccList = []

			while activeLabelNum < rounds:
		
				self.m_judgeClassifier.fit(targetNameFeatureIter, targetAuditorLabelIter)
				# self.m_clf.fit(targetNameFeatureIter, targetLabelIter) 

				exId = self.select_example(unlabeledExList) 
				
				transferLabel = targetTransferLabel[exId]
				# transferLabelFlag, transferLabel = self.get_transfer_flag(transferFeatureList, transferFlagList, exId)

				exLabel = self.m_targetLabel[exId]
				if transferLabel == exLabel:
					# print("correct transfer")
					targetNameFeatureIter = np.vstack((targetNameFeatureIter, self.m_targetNameFeature[exId]))
					targetAuditorLabelIter = np.hstack((targetAuditorLabelIter, 1.0))
					print(exId, transferLabel, exLabel, 1.0)
				else:
					# print("wrong transfer")
					targetNameFeatureIter = np.vstack((targetNameFeatureIter, self.m_targetNameFeature[exId]))
					targetAuditorLabelIter = np.hstack((targetAuditorLabelIter, 0.0))
					print(exId, transferLabel, exLabel, 0.0)

				labeledExList.append(exId)
				unlabeledExList.remove(exId)

				acc, precision, recall = self.get_pred_acc(targetNameFeatureTest, targetAuditorLabelTest, targetNameFeatureIter, targetAuditorLabelIter)
				totalAuditorAccList[cvIter].append(acc)
				totalAuditorPrecisionList[cvIter].append(acc)
				totalAuditorRecallList[cvIter].append(acc)
				# if activeLabelFlag:
				# 	humanAccList[cvIter].append(acc)
				queryIter += 1
				activeLabelNum += 1
				print("queryIter", queryIter)


			cvIter += 1      
		
		AuditorPrecisionFile = modelVersion+"_auditor_precision.txt"
		writeFile(totalAuditorPrecisionList, AuditorPrecisionFile)

		AuditorRecallFile = modelVersion+"_auditor_recall.txt"
		writeFile(totalAuditorRecallList, AuditorRecallFile)

		# AuditorAccFile = modelVersion+"_auditor_acc.txt"
		# writeFile(totalAuditorAccList, AuditorAccFile)

		totalACCFile = modelVersion+"_acc.txt"
		writeFile(totalAuditorAccList, totalACCFile)

		# humanACCFile = modelVersion+"_human_acc.txt"
		# writeFile(humanAccList, humanACCFile)

def writeFile(valueList, fileName):
	f = open(fileName, "w")
	for i in range(10):
		num4Iter = len(valueList[i])
		for j in range(num4Iter):
			f.write(str(valueList[i][j])+"\t")
		f.write("\n")
	f.close()

def data_analysis(sourceLabelList, targetLabelList):
	sourceLabelNum = len(sourceLabelList)
	sourceLabelMap = {}
	for sourceLabelIndex in range(sourceLabelNum):
		sourceLabelVal = int(sourceLabelList[sourceLabelIndex])

		if sourceLabelVal not in sourceLabelMap.keys():
			sourceLabelMap.setdefault(sourceLabelVal, 0.0)
		sourceLabelMap[sourceLabelVal] += 1.0

	sortedLabelList = sorted(sourceLabelMap.keys())

	# sortedLabelList = sorted(sourceLabelMap, key=sourceLabelMap.__getitem__, reverse=True)

	print("====source label distribution====")
	for label in sortedLabelList:
		print(label, sourceLabelMap[label], "--",)

	print("\n")
	targetLabelNum = len(targetLabelList)
	targetLabelMap = {}
	for targetLabelIndex in range(targetLabelNum):
		targetLabelVal = int(targetLabelList[targetLabelIndex])

		if targetLabelVal not in targetLabelMap.keys():
			targetLabelMap.setdefault(targetLabelVal, 0.0)
		targetLabelMap[targetLabelVal] += 1.0
	
	sortedLabelList = sorted(targetLabelMap.keys())

	# sortedLabelList = sorted(targetLabelMap, key=targetLabelMap.__getitem__, reverse=True)

	print("====target label distribution====")
	for label in sortedLabelList:
		print(label, targetLabelMap[label],"--",)
	print("\n")

if __name__ == "__main__":
	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../../data/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	targetLabel = tmp[:,-1]
	print 'target ---- class count of true labels of all ex:\n', ct(targetLabel)

	target_fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100

	input1 = np.genfromtxt("../../data/rice_hour_sdh", delimiter=",")
	fd1 = input1[:, 0:-1]
	target_fd = fd1
	target_label2 = input1[:,-1]

	input2 = np.genfromtxt("../../data/keti_hour_sum", delimiter=",")
	input3 = np.genfromtxt("../../data/sdh_hour_rice", delimiter=",")
	input2 = np.vstack((input2, input3))
	fd2 = input2[:, 0:-1]
	source_fd = fd2
	sourceLabel = input2[:,-1]

	print 'source ---- class count of true labels of all ex:\n', ct(sourceLabel)

	data_analysis(sourceLabel, targetLabel)

	al = _ActiveLearning(fold, rounds, source_fd, sourceLabel, target_fd, targetLabel, target_fn)

	al.run_CV()

