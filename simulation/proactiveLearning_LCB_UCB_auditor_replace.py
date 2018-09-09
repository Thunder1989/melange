"""
once we obtain a new annotation from strong oracle, we update the auditor's predictions for all previous weak labeled instances. 
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
import copy

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

modelName = "proactiveLearning_LCB_UCB_LCB_auditor_replace"
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

class _ProactiveLearning:

	def __init__(self, fold, rounds, featureMatrix, label, transferLabel):

		self.m_fold = fold
		self.m_rounds = rounds

		self.m_targetNameFeature = np.array(featureMatrix)
		self.m_targetLabel = np.array(label)

		self.m_transferLabel = np.array(transferLabel)

		self.m_randomForest = 0
		self.m_tao = 0

		self.m_alpha = 1

		self.m_lambda = 0.01
		self.m_A = 0
		self.m_AInv = 0
		self.m_cbRate = 0.002 ##0.05

		self.m_judgeClassifier = 0
		self.m_clf = 0

		self.m_labeledIDList = []
		self.m_strongLabeledIDList = []
		self.m_weakLabeledIDList = []
		self.m_unlabeledIDList = []

		self.m_replacedIDNum = 0
		self.m_correctionNum = 0

	def select_example(self):

		unlabeledIdScoreMap = {} ###unlabeledId:idscore

		unlabeledIdNum = len(self.m_unlabeledIDList)
		# unlabeledIdNum = len(unlabeled_list)

		labelNumMap = {} ###labelIndex:labelNum
		labelDensityMap = {} ###labelIndex:densityRatio

		labelIndexUnlabeledIdScoreMap = {} ### labelIndex:{unlabeledId:idscore}

		labelInstanceIDMap = {} ### labelIndex: [instanceIDList]

		for unlabeledIdIndex in range(unlabeledIdNum):
			unlabeledId = self.m_unlabeledIDList[unlabeledIdIndex]
			# unlabeledId = unlabeled_list[unlabeledIdIndex]
			# print("unlabeledId\t", unlabeledId)
			idScore = self.getBenefit4ActiveLearner(unlabeledId)
			unlabeledIdScoreMap[unlabeledId] = idScore
			
		# labelDensityMap /= unlabeledIdNum
		# sortedLabelIndexList = sorted(labelDensityMap, key=labelDensityMap.__getitem__, reverse=True)

		sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)
		# for labelIndex in sortedLabelIndexList:
			# sortedUnlabeledIdList = sorted(labelIndexUnlabeledIdScoreMap[labelIndex], key=labelIndexUnlabeledIdScoreMap[labelIndex].__getitem__, reverse=True)

		return sortedUnlabeledIdList[0]

	def getBenefit4ActiveLearner(self, exId):
		unlabeledId = exId
		labelPredictProb = self.m_clf.predict_proba(self.m_targetNameFeature[unlabeledId].reshape(1, -1))[0]

		labelIndexMap = {} ##labelIndex: labelProb
		labelNum = len(labelPredictProb)
		for labelIndex in range(labelNum):
			labelIndexMap.setdefault(labelIndex, labelPredictProb[labelIndex])

		sortedLabelIndexList = sorted(labelIndexMap, key=labelIndexMap.__getitem__, reverse=True)

		maxLabelIndex = sortedLabelIndexList[0]
		subMaxLabelIndex = sortedLabelIndexList[1]

		selectCB = self.get_select_confidence_bound(unlabeledId)

		coefDiff = 0
		if labelNum == 2:
			# coefDiff = np.dot(self.clf.coef_, self.fn[unlabeledId])
			print("error")
			print(debug)
		else:
			maxCoef = self.m_clf.coef_[maxLabelIndex]
			subMaxCoef = self.m_clf.coef_[subMaxLabelIndex]
			coefDiff = np.dot(maxCoef, self.m_targetNameFeature[unlabeledId])-np.dot(subMaxCoef, self.m_targetNameFeature[unlabeledId])
		
		LCB = coefDiff-2*0.002*selectCB
		idScore = 1-LCB

		return idScore

	def get_pred_acc(self, targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter):

		# targetNameFeatureTrain = self.m_targetNameFeature[labeledIdList]
		# targetLabelTrain = self.m_targetLabel[labeledIdList]
		
		self.m_clf.fit(targetNameFeatureIter, targetLabelIter)
		targetLabelPreds = self.m_clf.predict(targetNameFeatureTest)

		acc = accuracy_score(targetLabelTest, targetLabelPreds)
		# print("acc\t", acc)
		# print debug
		return acc

	def get_base_learners(self):
		self.m_randomForest = RFC(n_estimators=100, criterion='entropy', random_state=3)

		self.m_randomForest.fit(self.m_sourceDataFeature, self.m_sourceLabel)

	def init_confidence_bound(self, featureDim, labeledExList, unlabeledExList):

		# self.m_labeledIDList = labeledExList

		self.m_strongLabeledIDList = labeledExList
		self.m_unlabeledIDList = unlabeledExList
		self.m_weakLabeledIDList = []

		self.m_selectA = self.m_lambda*np.identity(featureDim)
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

		self.m_judgeA = self.m_lambda*np.identity(featureDim)
		self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

		self.m_replacedIDNum = 0
		self.m_correctionNum = 0

	def update_select_confidence_bound_addID(self, exId):
		# print("updating select cb", exId)
		# self.m_labeledIDList.append(exId)
		# self.m_strongLabeledIDList.append(exId)
		# self.m_weakLabeledIDList.append(exId)
		self.m_unlabeledIDList.remove(exId)
		self.m_selectA += np.outer(self.m_targetNameFeature[exId], self.m_targetNameFeature[exId])
		self.m_selectAInv = np.linalg.inv(self.m_selectA)

	def update_select_confidence_bound_removeID(self, exId):
		# self.m_labeledIDList.remove(exId)
		# self.m_weakLabeledIDList
		self.m_selectA -= np.outer(self.m_targetNameFeature[exId], self.m_targetNameFeature[exId])
		self.m_selectAInv = np.linalg.inv(self.m_selectA)
		self.m_unlabeledIDList.append(exId)

	def update_judge_confidence_bound(self, exId):
		# self.m_weakLabeledIDList.remove(exId)
		# print("updating judge cb", exId)
		self.m_judgeA += np.outer(self.m_targetNameFeature[exId], self.m_targetNameFeature[exId])
		self.m_judgeAInv = np.linalg.inv(self.m_judgeA)

	def get_select_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.m_targetNameFeature[exId], self.m_selectAInv), self.m_targetNameFeature[exId]))

		return CB

	def get_judge_confidence_bound(self, exId):
		CB = np.sqrt(np.dot(np.dot(self.m_targetNameFeature[exId], self.m_judgeAInv), self.m_targetNameFeature[exId]))

		return CB

	def get_judgeClassifier_prob(self, judgeParam, feature, CB):
		rawProb = np.dot(judgeParam, np.transpose(feature))

		CBProb = sigmoid(rawProb-self.m_cbRate*CB)

		return CBProb

	def updateAuditor(self, transferFeatureList, transferFlagList):
		if len(np.unique(transferFlagList)) > 1:
			self.m_judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))

	def get_transfer_flag(self, transferFeatureList, transferFlagList, exID):
		judgeProbThreshold = 0.5

		if len(np.unique(transferFlagList)) < 2:
			return False

		CBProb = self.getTransferProb(exID)

		if CBProb > judgeProbThreshold:
			return True
		else:
			return False

	def getTransferProb(self, exId):

		CB = self.get_judge_confidence_bound(exId)

		CBProb = self.get_judgeClassifier_prob(self.m_judgeClassifier.coef_, self.m_targetNameFeature[exId].reshape(1, -1), CB)

		return CBProb

	### transferFeatureList, transferFlagList stores only features of instances labeled by strong oracle
	def updateWeakLabeledIDList(self, transferFeatureList, transferFlagList):
		weakLabeledNum = len(self.m_weakLabeledIDList)
		weakLabeledIDListTemp = copy.copy(self.m_weakLabeledIDList)

		unlabeledIDAuditorProbMap = {} ### id: prob
		unlabeledIDActiveBenefitMap = {} ### id: benefit

		if len(np.unique(transferFlagList)) < 2:
			return 

		unlabeledIdNum = len(self.m_unlabeledIDList)
		for unlabelIndex in range(unlabeledIdNum):
			unlabeledID = self.m_unlabeledIDList[unlabelIndex]

			unlabeledIDAuditorProb = self.getTransferProb(unlabeledID)
			unlabeledIDAuditorProbMap.setdefault(unlabeledID, unlabeledIDAuditorProb)

			activeLearnerBenefit4UnlabeledID = self.getBenefit4ActiveLearner(unlabeledID)
			unlabeledIDActiveBenefitMap.setdefault(unlabeledID, activeLearnerBenefit4UnlabeledID)

		falsePosLabeledIDList = []
		for weakLabeledIndex in range(weakLabeledNum):
			weakLabeledID = weakLabeledIDListTemp[weakLabeledIndex]
			transferFlag = self.get_transfer_flag(transferFeatureList, transferFlagList, weakLabeledID)

			if not transferFlag:
				self.m_correctionNum += 1
				self.m_weakLabeledIDList.remove(weakLabeledID)

				self.getReplacedID(unlabeledIDAuditorProbMap, unlabeledIDActiveBenefitMap, weakLabeledID, transferFlagList)

				falsePosLabeledIDList.append(weakLabeledID)
				
		for falsePosLabeledID in falsePosLabeledIDList:
			print("removing", falsePosLabeledID)
			self.update_select_confidence_bound_removeID(falsePosLabeledID)

	def getReplacedID(self, unlabeledIDAuditorProbMap, unlabeledIDActiveBenefitMap, falsePosLabeledID, transferFlagList):

		falsePosLabeledIDProb = self.getTransferProb(falsePosLabeledID)
		activeLearnerBenefit4FalsePosLabeledID = self.getBenefit4ActiveLearner(falsePosLabeledID)

		replacedIDScoreMap = {} ##id: score
		unlabeledIdNum = len(self.m_unlabeledIDList)
		for unlabelIndex in range(unlabeledIdNum):
			unlabeledID = self.m_unlabeledIDList[unlabelIndex]
			if unlabeledIDAuditorProbMap[unlabeledID] > falsePosLabeledIDProb:
				if unlabeledIDAuditorProbMap[unlabeledID] > 0.5:
					if unlabeledIDActiveBenefitMap[unlabeledID] > activeLearnerBenefit4FalsePosLabeledID:
						replacedIDScoreMap.setdefault(unlabeledID, unlabeledIDAuditorProbMap[unlabeledID]*unlabeledIDActiveBenefitMap[unlabeledID])

		sortedUnlabeledIDList = sorted(replacedIDScoreMap, key=replacedIDScoreMap.__getitem__, reverse=True)

		if len(sortedUnlabeledIDList) > 0:
			replacedID = sortedUnlabeledIDList[0]
			print("replacing", replacedID)
			self.m_replacedIDNum += 1
			self.update_select_confidence_bound_addID(replacedID)
			self.m_weakLabeledIDList.append(replacedID) 

	def getAuditorMetric(self, transferFeatureList, transferFlagList, transferFeatureTest, transferLabelTest, targetLabelTest):
		acc = 0.0
		if len(np.unique(transferFlagList)) > 1:
			# self.m_judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))

			auditorLabelTest = (transferLabelTest==targetLabelTest)

			predictAuditorLabelTest = self.m_judgeClassifier.predict(transferFeatureTest)

			acc = accuracy_score(auditorLabelTest, predictAuditorLabelTest)

		return acc 


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
		totalAccList = [[] for i in range(10)]
		humanAccList = [[] for i in range(10)]

		# self.get_base_learners()

		# correctTransferRatioList = []
		totalTransferNumList = []
		totalReplaceNumList = []
		totalCorrectionNumList = []
		# correctUntransferRatioList = []

		totalAuditorPrecisionList = []
		totalAuditorRecallList = []
		totalAuditorAccList = []

		for foldIndex in range(foldNum):
			
			# self.clf = LinearSVC(random_state=3)

			self.m_clf = LR(multi_class="multinomial", solver='lbfgs',random_state=3, fit_intercept=False)
			self.m_judgeClassifier = LR(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)

			targetNameFeatureTrain = self.m_targetNameFeature[train]
			targetLabelTrain = self.m_targetLabel[train]

			targetNameFeatureTest = self.m_targetNameFeature[test]
			targetLabelTest = self.m_targetLabel[test]

			transferLabelTest = self.m_transferLabel[test]

			initExList = []
			random.seed(101)
			initExList = random.sample(train, 3)

			targetNameFeatureInit = self.m_targetNameFeature[initExList]
			targetLabelInit = self.m_targetLabel[initExList]

			print("initExList\t", initExList, targetLabelInit)

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
			self.init_confidence_bound(featureDim, labeledExList, unlabeledExList)

			targetNameFeatureIter = targetNameFeatureInit
			targetLabelIter = targetLabelInit

			correctTransferLabelNum = 0.0
			wrongTransferLabelNum = 0.0
			correctUntransferLabelNum = 0.0
			wrongUntransferLabelNum = 0.0

			auditorPrecisionList = []
			auditorRecallList = []
			auditorAccList = []

			while activeLabelNum < rounds:

				# targetNameFeatureIter = self.m_targetNameFeature[labeledExList]
				# targetLabelIter = self.m_targetLabel[labeledExList]

				self.m_clf.fit(targetNameFeatureIter, targetLabelIter) 

				exId = self.select_example() 
				self.update_select_confidence_bound_addID(exId)

				# print(idx)
				activeLabelFlag = False
				self.updateAuditor(transferFeatureList, transferFlagList)
				transferLabelFlag = self.get_transfer_flag(transferFeatureList, transferFlagList, exId)

				exLabel = -1

				transferLabel = self.m_transferLabel[exId]
				if transferLabelFlag:
					self.m_weakLabeledIDList.append(exId)
					print("queryIter\t", queryIter, exId)
					transferLabelNum += 1.0
					activeLabelFlag = False
					
					exLabel = transferLabel

					if exLabel == self.m_targetLabel[exId]:
						correctTransferLabelNum += 1.0
					else:
						wrongTransferLabelNum += 1.0
						# print("query iteration", queryIter, "error transfer label\t", exLabel, "true label", self.m_targetLabel[exId])
				else:
					self.m_strongLabeledIDList.append(exId)
					self.update_judge_confidence_bound(exId)
					activeLabelNum += 1.0
					activeLabelFlag = True

					exLabel = self.m_targetLabel[exId]

					if transferLabel == exLabel:
						correctUntransferLabelNum += 1.0
						transferFlagList.append(1.0)
						transferFeatureList.append(self.m_targetNameFeature[exId])
					else:
						wrongUntransferLabelNum += 1.0
						transferFlagList.append(0.0)
						transferFeatureList.append(self.m_targetNameFeature[exId])

					self.updateAuditor(transferFeatureList, transferFlagList)
					auditorAcc = self.getAuditorMetric(transferFeatureList, transferFlagList, targetNameFeatureTest, transferLabelTest, targetLabelTest)
					# print("auditorAcc", auditorAcc)

					# auditorPrecisionList.append(auditorPrecision)
					# auditorRecallList.append(auditorRecall)
					auditorAccList.append(auditorAcc)

					self.updateWeakLabeledIDList(transferFeatureList, transferFlagList)

					if len(self.m_weakLabeledIDList) > 0:
						targetNameFeatureIter = np.vstack((self.m_targetNameFeature[self.m_strongLabeledIDList], self.m_targetNameFeature[self.m_weakLabeledIDList]))
						targetLabelIter = np.hstack((self.m_targetLabel[self.m_strongLabeledIDList], self.m_transferLabel[self.m_weakLabeledIDList]))

					else:
						targetNameFeatureIter = self.m_targetNameFeature[self.m_strongLabeledIDList]
						targetLabelIter = self.m_targetLabel[self.m_strongLabeledIDList]
				# labeledExList.append(exId)
				# unlabeledExList.remove(exId)

				acc = self.get_pred_acc(targetNameFeatureTest, targetLabelTest, targetNameFeatureIter, targetLabelIter)
				totalAccList[cvIter].append(acc)
				if activeLabelFlag:
					humanAccList[cvIter].append(acc)
				queryIter += 1

			# totalAuditorPrecisionList.append(auditorPrecisionList)
			# totalAuditorRecallList.append(auditorRecallList)
			totalAuditorAccList.append(auditorAccList)

			# correctUntransferRatio = correctUntransferLabelNum*1.0
			# correctUntransferRatioList.append(correctUntransferRatio)
			# print("correctUntransferRatio\t", correctUntransferRatio)

			# correctTransferRatio = correctTransferLabelNum*1.0/transferLabelNum
			# print("transferLabelNum\t", transferLabelNum, "correct transfer ratio\t", correctTransferRatio)
			# correctTransferRatioList.append(correctTransferRatio)
			transferLabelNum = len(self.m_weakLabeledIDList)
			totalTransferNumList.append(transferLabelNum)

			replaceNum = self.m_replacedIDNum
			totalReplaceNumList.append(replaceNum)

			totalCorrectionNumList.append(self.m_correctionNum)

			cvIter += 1      
		
		print("transfer num\t", np.mean(totalTransferNumList), np.sqrt(np.var(totalTransferNumList)))
		print("replace num\t", np.mean(totalReplaceNumList), np.sqrt(np.var(totalReplaceNumList)))
		print("correction num\t", np.mean(totalCorrectionNumList), np.sqrt(np.var(totalCorrectionNumList)))
		# print("correct ratio\t", np.mean(correctTransferRatioList), np.sqrt(np.var(correctTransferRatioList)))
		# print("untransfer correct ratio\t", np.mean(correctUntransferRatioList), np.sqrt(np.var(correctUntransferRatioList)))

		# AuditorPrecisionFile = modelVersion+"_auditor_precision.txt"
		# writeFile(totalAuditorPrecisionList, AuditorPrecisionFile)

		# AuditorRecallFile = modelVersion+"_auditor_recall.txt"
		# writeFile(totalAuditorRecallList, AuditorRecallFile)

		AuditorAccFile = modelVersion+"_auditor_acc.txt"
		writeFile(totalAuditorAccList, AuditorAccFile)

		totalACCFile = modelVersion+"_acc.txt"
		writeFile(totalAccList, totalACCFile)

		humanACCFile = modelVersion+"_human_acc.txt"
		writeFile(humanAccList, humanACCFile)

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

	# raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('./selectedNameFeature4Label_5types.txt').readlines()]

	# raw_pt = [i.strip().split('\t')[:-1] for i in open().readlines()]

	f = open('./simulatedFeatureLabel_500_100_10.txt')
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
	# tmp = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
	# label = tmp[:,-1]
	# selectedInstanceIDList = [float(i.strip()) for i in open('selectedID4Label.txt').readlines()]
	f = open('./simulatedTransferLabel_500_100_10.txt')
	transferLabel = []

	for rawLine in f:
		if "truelabel" in rawLine:
			continue

		featureLine = rawLine.strip().split("\t")
	
		transferLabelVal = float(featureLine[1])

		transferLabel.append(transferLabelVal)

	f.close()

	print("number of types", len(set(label)))
	print 'class count of true labels of all ex:\n', ct(transferLabel)

	fold = 10
	rounds = 100

	al = _ProactiveLearning(fold, rounds, featureMatrix, label, transferLabel)

	al.run_CV()