"""
active transfer learning we use the lower bound to estimate whether we should trust the classifer or not. LCB and we try K times to see whether it could past. If K times pass the judge classifier, we use the instance which pass the judge classifier. Otherwise, we use the first one to ask for human label. 
We use the entropy times the distance to compute the score and plus the CB
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

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

class transferActiveLearning:

	def __init__(self, fold, rounds, source_fd, source_label, target_fd, target_label, target_fn):

		self.fold = fold
		self.rounds = rounds
		self.acc_sum = [[] for i in xrange(self.rounds)] #acc per iter for each fold

		self.m_source_fd = source_fd
		self.m_source_label = source_label

		self.m_target_fd = target_fd

		self.m_target_fn = target_fn
		self.m_target_label = target_label

		self.rf = 0

		self.tao = 0
		self.alpha_ = 1

		self.clf = 0
		self.ex_id = dd(list)

		self.judgeClassifier = 0
		self.m_cbRate = 0.05

### track top k examples try to pass judgeClassifier
		self.m_topKExList = []
		self.m_topK = 5

	def update_tao(self, al_tl_fn_train, al_tl_label_train):

		dist_inter = []
		fn_train_tao = np.array(al_tl_fn_train)
		label_train_tao = np.array(al_tl_label_train)

		indexList_tao = [i for i in range(len(label_train_tao))]

		pair = list(itertools.combinations(indexList_tao,2))

		for p in pair:
			if label_train_tao[p[0]] != label_train_tao[p[1]]:
				d = np.linalg.norm(fn_train_tao[p[0]]-fn_train_tao[p[1]])
				dist_inter.append(d)

		try:
			self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2
		except Exception as e:
			self.tao = self.tao

	def update_pseudo_set(self, new_ex_id, new_ex_label, cluster_id, p_idx, p_label, p_dist):

		tmp = []
		idx_tmp=[]
		label_tmp=[]

		#re-visit exs removed on previous itr with the new tao
		for i,j in zip(p_idx,p_label):

			if p_dist[i] < self.tao:
				idx_tmp.append(i)
				label_tmp.append(j)
			else:
				p_dist.pop(i)
				tmp.append(i)

		p_idx = idx_tmp
		p_label = label_tmp

		#added exs to pseudo set
		for ex in self.ex_id[cluster_id]:

			if ex == new_ex_id:
				continue
			d = np.linalg.norm(self.m_target_fn[ex]-self.m_target_fn[new_ex_id])

			if d < self.tao:
				p_dist[ex] = d
				p_idx.append(ex)
				# p_label.append(self.m_target_label[new_ex_id])
				p_label.append(new_ex_label)
			else:
				tmp.append(ex)

		if not tmp:
			self.ex_id.pop(cluster_id)
		else:
			self.ex_id[cluster_id] = tmp

		return p_idx, p_label, p_dist
	
	def select_example(self, labeled_set):

		sub_pred = dd(list) #Mn predicted labels for each cluster
		idx = 0
		# topKidxList = []
		# topKcidxList = []

		idxScore = {}
		idxCluster = {} ##clusterID:idx

		for k,v in self.ex_id.items():
			sub_pred[k] = self.clf.predict(self.m_target_fn[v]) #predict labels for cluster learning set

		#entropy-based cluster selection
		rank = {}
		for k,v in sub_pred.items():
			count = ct(v).values()
			count[:] = [i/float(max(count)) for i in count]
			H = np.sum(-p*math.log(p,2) for p in count if p!=0)
			rank.setdefault(k, H)
		# rank = sorted(rank, key=lambda x: x[-1], reverse=True)

		rankedClusterList = sorted(rank, key=rank.__getitem__, reverse=True)
		if not rankedClusterList:
			raise ValueError('no clusters found in this iteration!')

		for k, v in self.ex_id.items():
			idNum = len(v)
			for idIndex in range(idNum):
				idx = v[idIndex]
				idxScore[idx] = rank[k]
				idxCluster[idx] = k

				idxScore[idx] *= self.getConfidenceBound(idx)

		for k, v in self.ex_id.items():
			c_idx = k
		# c_idx = rank[0][0] #pick the 1st cluster on the rank, ordered by label entropy
			c_ex_id = self.ex_id[c_idx] #examples in the cluster picked
			sub_label = sub_pred[c_idx] #used when choosing cluster by H
			sub_fn = self.m_target_fn[c_ex_id]

			#sub-cluster the cluster
			c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10, random_state=3)
			c_.fit(sub_fn)
			dist = np.sort(c_.transform(sub_fn))

			ex_ = dd(list)
			maxExNum = 0
			totalLen = 0

			for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
				ex_[i].append([j,l,k[0]])
				if j not in idxScore.keys():
					print("error\t", j)
				if k[0] == 0.0:
					idxScore[j] *= 1e8
				else:
					idxScore[j] *= 1/k[0]

			# for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
			# 	ex_[i] = sorted(j, key=lambda x: x[-1])
			# 	totalLen += len(ex_[i])
			# 	if len(ex_[i]) > maxExNum:
			# 		maxExNum = len(ex_[i])

		candidateIdxList = sorted(idxScore, key=idxScore.__getitem__, reverse=True)


		# for subIndex in range(maxExNum):
		# 	for k,v in ex_.items():
				
		# 		if subIndex >= len(ex_[k]):
		# 				continue
		# 		if v[subIndex][0] not in labeled_set: 
		# 			candidateIdxList.append(v[subIndex][0])

		# print("totalLen\t", totalLen, len(candidateIdxList))
		candidateClusterList = []

		if len(candidateIdxList) < self.m_topK:
			print("less than topk\t", self.m_topK, len(candidateIdxList))
			for idIndex in range(len(candidateIdxList)):
				idx = candidateIdxList[idIndex]
				candidateClusterList.append(idxCluster[idx])
			return candidateIdxList, candidateClusterList
		else:
			for idIndex in range(self.m_topK):
				idx = candidateIdxList[idIndex]
				candidateClusterList.append(idxCluster[idx])
			return candidateIdxList[:self.m_topK], candidateClusterList

			# if v[subIndex][0] not in labeled_set: #find the first unlabeled ex

			# 	idx = v[subIndex][0]

			# 	if len(topKidxList) > self.m_topK:
			# 		break

			# 	topKidxList.append(idx)
			# 	topKcidxList.append(c_idx)



				# c_ex_id.remove(idx) #update the training set by removing selected ex id

				# self.ex_id[c_idx] = c_ex_id

				# if len(c_ex_id) == 0:
				# 	self.ex_id.pop(c_idx)
				# else:
				# 	self.ex_id[c_idx] = c_ex_id
				# break

		# return topKidxList, topKcidxList

	def get_pred_acc(self, fn_test, label_test, al_tl_fn_train, al_tl_label_train, pseudo_set, pseudo_label):

		fn_train_pred = []
		label_train_pred = []

		if not pseudo_set:
			fn_train_pred = np.array(al_tl_fn_train)
			label_train_pred = np.array(al_tl_label_train)
		else:
			fn_train_pred = self.m_target_fn[pseudo_set]
			fn_train_pred = np.vstack((fn_train_pred, al_tl_fn_train))

			# print("pseudo_label\t", pseudo_label, al_tl_label_train)
			label_train_pred = np.hstack((pseudo_label, al_tl_label_train))

		self.clf.fit(fn_train_pred, label_train_pred)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)

		return acc

	def get_base_learners(self):
		self.rf = RFC(n_estimators=100, criterion='entropy', random_state=3)
		self.rf.fit(self.m_source_fd, self.m_source_label)
	
	def initConfidenceBound(self, _lambda, featureDim):
		self.m_A = _lambda*np.identity(featureDim)

		self.m_AInv = np.linalg.inv(self.m_A)

	def updateConfidenceBound(self, idx):
		self.m_A += np.outer(self.m_target_fn[idx], self.m_target_fn[idx])
		self.m_AInv = np.linalg.inv(self.m_A)

	def getConfidenceBound(self, idx):

		CB = np.sqrt(np.dot(np.dot(self.m_target_fn[idx], self.m_AInv), self.m_target_fn[idx]))

		return CB

	def getJudgeProb(self, judgeParam, feature, CB):
		rawProb = np.dot(judgeParam, np.transpose(feature))
		judgeProbThreshold = 0.5
		if sigmoid(rawProb-self.m_cbRate*CB) > judgeProbThreshold:
			return True
		else:
			return False

	def topKTransferOrNot(self, transferFeatureList, transferFlagList, idxList):
		transferLabelFlag = False
		idxNum = len(idxList)

		transferIndex = 0
		transferLabel = 0

		for idxIndex in range(idxNum):
			idx = idxList[idxIndex]
			transferFlag, predLabel = self.transferOrNot(transferFeatureList, transferFlagList, idx)

			if transferFlag == True:
				transferIndex = idxIndex
				transferLabel = predLabel
			
				return transferIndex, transferFlag, transferLabel

		idx = idxList[0]
		transferIndex = 0
		transferFlag, transferLabel = self.transferOrNot(transferFeatureList, transferFlagList, idx)

		return transferIndex, transferFlag, transferLabel

	def transferOrNot(self, transferFeatureList, transferFlagList, idx):
		
		predLabel = self.rf.predict(self.m_target_fd[idx].reshape(1, -1))[0]

		if len(np.unique(transferFlagList)) > 1:
			self.judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))
		else:
			return False, predLabel

		CB = self.getConfidenceBound(idx) 

		# LCB = self.judgeClassifier.coef_ - self.m_cbRate*CB
		# UCB = self.judgeClassifier.coef_ + self.m_cbRate*CB

		transferFlag = self.getJudgeProb(self.judgeClassifier.coef_, self.m_target_fn[idx].reshape(1, -1), CB)
		if transferFlag:
			return True, predLabel
		else:
			return False, predLabel

	def run_CV(self):

		totalInstanceNum = len(self.m_target_label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		totalTransferNumList = []
		correctTransferRatioList = []
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
		activeAccList = [[] for i in range(10)]
		totalAccList = [[] for i in range(10)]

		self.get_base_learners()

		for foldIndex in range(foldNum):
			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			# print train, test
			self.clf = LinearSVC(random_state=3)
			self.judgeClassifier = LR(random_state=3)
			print("cvIter...\t",cvIter)
		
			fn_train = self.m_target_fn[train]
			label_train = self.m_target_label[train]
			fd_train = self.m_target_fd[train]

			fn_test = self.m_target_fn[test]
			label_test = self.m_target_label[test]
			fd_test = self.m_target_fd[test]

			###transfer learning

			class_ = np.unique(self.m_source_label)

			c = KMeans(init='k-means++', n_clusters=28, n_init=10, random_state=3)
			c.fit(fn_train)
			dist = np.sort(c.transform(fn_train))

			ex = dd(list) #example id, distance to centroid
			self.ex_id = dd(list) #example id for each C
			ex_N = [] # num of examples in each C
			for i,j,k in zip(c.labels_, train, dist):
				ex[i].append([j,k[0]])
				self.ex_id[i].append(int(j))
			for i,j in ex.items():
				ex[i] = sorted(j, key=lambda x: x[-1])
				ex_N.append([i,len(ex[i])])
			ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True)

			km_idx = []
			p_idx = []
			p_label = []
			p_dist = dd()
			#first batch of exs: pick centroid of each cluster, and cluster visited based on its size
			
			###only active label count for the comparison
			ctr = 0

			activeLabelNum = 0

			transferLearnerThreshold = 0.5

			al_tl_label_train = []
			al_tl_fn_train = []

			transferFeatureList = []
			transferFlagList = []
			transferLabelNum = 0.0

			correctTransferLabelNum = 0.0

			queryIteration = 0 

			_lambda = 0.01
			featureDim = len(fn_train[0])
			self.initConfidenceBound(_lambda, featureDim)

			clusterNum = len(ex_N)
			clusterIndex = 0
			while clusterIndex < 3:
			# for ee in ex_N:
				ee = ex_N[clusterIndex]
				clusterIndex += 1
				c_idx = ee[0] #cluster id
				idx = ex[c_idx][0][0] #id of ex closest to centroid of cluster

				# judgeFailTimes = 0
				# judgeFailIDList = []
				# for judgeFailIndex in range(self.m_topK):
				transferLabelFlag, label_transfer = self.transferOrNot(transferFeatureList, transferFlagList, idx)
			

				activeLabelFlag = False
				label_idx = -1
				if transferLabelFlag:

					transferLabelNum += 1.0
					activeLabelFlag = False
					label_idx = label_transfer
					# label_idx = self.m_target_label[idx]
					al_tl_label_train.append(label_idx)
					al_tl_fn_train.append(self.m_target_fn[idx])

					if label_idx == self.m_target_label[idx]:
						correctTransferLabelNum += 1.0
						print('transfer correct')
						print(debug)
					else:
						print(queryIteration, "error transfer label\t", label_transfer, "true label", self.m_target_label[idx])

				else:
					#active learning
					activeLabelNum += 1.0
					activeLabelFlag = True

					self.updateConfidenceBound(idx)				

					label_idx = self.m_target_label[idx]
					al_tl_label_train.append(label_idx)
					al_tl_fn_train.append(self.m_target_fn[idx])

					if label_transfer == label_idx:
						transferFlagList.append(1.0)
						transferFeatureList.append(self.m_target_fn[idx])
					else:
						transferFlagList.append(0.0)
						transferFeatureList.append(self.m_target_fn[idx])

				queryIteration += 1
				km_idx.append(idx)
				ctr+=1

				tmp = self.ex_id[c_idx]
				tmp.remove(idx)

				if len(tmp) == 0:
					self.ex_id.pop(c_idx)
				else:
					self.ex_id[c_idx] = tmp

				if ctr<3:
					
					continue

				self.update_tao(al_tl_fn_train, al_tl_label_train)
				# self.update_tao(km_idx)

				p_idx, p_label, p_dist = self.update_pseudo_set(idx, label_idx, c_idx, p_idx, p_label, p_dist)

				acc = self.get_pred_acc(fn_test, label_test, al_tl_fn_train, al_tl_label_train, p_idx, p_label)
					
				# acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
				if activeLabelFlag:
					activeAccList[cvIter].append(acc)
				totalAccList[cvIter].append(acc)

			cl_id = [] #track cluster id on each iter
			ex_al = [] #track ex added on each iter
			fn_test = self.m_target_fn[test]
			label_test = self.m_target_label[test]
			# for rr in range(ctr, rounds):

			while activeLabelNum < rounds:

				fn_train_iter = []
				label_train_iter = []
				queryIteration += 1
				# print("queryIteration\t", queryIteration)

				if not p_idx:
					fn_train_iter = np.array(al_tl_fn_train)
					label_train_iter = np.array(al_tl_label_train)
				else:
					fn_train_iter = self.m_target_fn[p_idx]
					label_train_iter = p_label
					fn_train_iter = np.vstack((fn_train_iter, np.array(al_tl_fn_train)))
					label_train_iter = np.hstack((label_train_iter, al_tl_label_train))

				self.clf.fit(fn_train_iter, label_train_iter) 
				# print(km_idx)                         
				topKidxList, topKcidxList = self.select_example(km_idx)  

				activeLabelFlag = False

				transferIndex, transferLabelFlag, label_transfer = self.topKTransferOrNot(transferFeatureList, transferFlagList, topKidxList)

				idx = topKidxList[transferIndex]
				c_idx = topKcidxList[transferIndex]
				
				tmp = self.ex_id[c_idx]
				tmp.remove(idx)

				if len(tmp) == 0:
					self.ex_id.pop(c_idx)
				else:
					self.ex_id[c_idx] = tmp

				label_idx = -1
				if transferLabelFlag:
					##transfer
					transferLabelNum += 1.0
					activeLabelFlag = False
					label_idx = label_transfer
					# label_idx = self.m_target_label[idx]
					al_tl_label_train.append(label_idx)
					al_tl_fn_train.append(self.m_target_fn[idx])

					if label_idx == self.m_target_label[idx]:
						correctTransferLabelNum += 1.0
					else:
						print(queryIteration, "error transfer label\t", label_transfer, "true label", self.m_target_label[idx])
				else:
					##active 
					activeLabelNum += 1.0
					activeLabelFlag = True
					label_idx = self.m_target_label[idx]

					al_tl_label_train.append(self.m_target_label[idx])
					al_tl_fn_train.append(self.m_target_fn[idx])

					self.updateConfidenceBound(idx)				

					if label_transfer == label_idx:
						transferFlagList.append(1.0)
						transferFeatureList.append(self.m_target_fn[idx])
					else:
						transferFlagList.append(0.0)
						transferFeatureList.append(self.m_target_fn[idx])
	
				km_idx.append(idx)
				cl_id.append(c_idx) #track picked cluster id on each iteration
				# ex_al.append([rr,key,v[0][-2],self.m_target_label[idx],raw_pt[idx]]) #for debugging
				# self.update_tao(km_idx)

				self.update_tao(al_tl_fn_train, al_tl_label_train)
			
				p_idx, p_label, p_dist = self.update_pseudo_set(idx, label_idx, c_idx, p_idx, p_label, p_dist)
				
				acc = self.get_pred_acc(fn_test, label_test, al_tl_fn_train, al_tl_label_train, p_idx, p_label)

				# acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)

				if activeLabelFlag:
					activeAccList[cvIter].append(acc)
				totalAccList[cvIter].append(acc)

			correctRatio = correctTransferLabelNum*1.0/transferLabelNum
			print("transferLabelNum\t", transferLabelNum, "correct ratio\t", correctRatio)
			correctTransferRatioList.append(correctRatio)
			totalTransferNumList.append(transferLabelNum)
			# print(debug)
			cvIter += 1

		print("transfer num\t", np.mean(totalTransferNumList), np.sqrt(np.var(totalTransferNumList)))
		print("correct ratio\t", np.mean(correctTransferRatioList), np.sqrt(np.var(correctTransferRatioList)))

		f = open("proactiveLearning_2_total.txt", "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

		f = open("proactiveLearning_2_human.txt", "w")
		for i in range(10):
			totalAlNum = len(activeAccList[i])
			for j in range(totalAlNum):
				f.write(str(activeAccList[i][j])+"\t")
			f.write("\n")
		f.close()

if __name__ == "__main__":
	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}


	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
	target_label = tmp[:,-1]
	print 'target ---- class count of true labels of all ex:\n', ct(target_label)

	target_fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100

	input1 = np.genfromtxt("../data/rice_hour_sdh", delimiter=",")
	fd1 = input1[:, 0:-1]
	target_fd = fd1
	target_label2 = input1[:,-1]


	input2 = np.genfromtxt("../data/keti_hour_sum", delimiter=",")
	input3 = np.genfromtxt("../data/sdh_hour_rice", delimiter=",")
	input2 = np.vstack((input2, input3))
	fd2 = input2[:, 0:-1]
	source_fd = fd2
	source_label = input2[:,-1]

	print 'source ---- class count of true labels of all ex:\n', ct(source_label)

	al = transferActiveLearning(fold, rounds, source_fd, source_label, target_fd, target_label, target_fn)

	al.run_CV()