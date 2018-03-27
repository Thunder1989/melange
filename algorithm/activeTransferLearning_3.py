"""
In this setting, transfer learning serves as a weak oracle and is utilized in the framework of active learning
cluster together and ask for both active learning and transfer learning
we use judgeClassifier to help us select instance when querying for label
When the number of labels go beyond a threshold, we seek transfer learn for help.
output those labels of transfer learning
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
		# self.fn = fn
		# self.m_target_label = label

		self.bl = []

		self.tao = 0
		self.alpha_ = 1

		self.clf = LinearSVC()
		self.ex_id = dd(list)


	def update_tao(self, al_tl_fn_train, al_tl_label_train):

		dist_inter = []
		fn_train_tao = np.array(al_tl_fn_train)
		label_train_tao = np.array(al_tl_label_train)

		indexList_tao = [i for i in range(len(label_train_tao))]

		pair = list(itertools.combinations(indexList_tao,2))

		for p in pair:

			d = np.linalg.norm(fn_train_tao[p[0]]-fn_train_tao[p[1]])
			if label_train_tao[p[0]] != label_train_tao[p[1]]:
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

		for k,v in self.ex_id.items():
			sub_pred[k] = self.clf.predict(self.m_target_fn[v]) #predict labels for cluster learning set

		#entropy-based cluster selection
		rank = []
		for k,v in sub_pred.items():
			count = ct(v).values()
			count[:] = [i/float(max(count)) for i in count]
			H = np.sum(-p*math.log(p,2) for p in count if p!=0)
			rank.append([k,len(v),H])
		rank = sorted(rank, key=lambda x: x[-1], reverse=True)

		if not rank:
			raise ValueError('no clusters found in this iteration!')        

		c_idx = rank[0][0] #pick the 1st cluster on the rank, ordered by label entropy
		c_ex_id = self.ex_id[c_idx] #examples in the cluster picked
		sub_label = sub_pred[c_idx] #used when choosing cluster by H
		sub_fn = self.m_target_fn[c_ex_id]

		#sub-cluster the cluster
		c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
		c_.fit(sub_fn)
		dist = np.sort(c_.transform(sub_fn))

		ex_ = dd(list)
		for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
			ex_[i].append([j,l,k[0]])
		for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
			ex_[i] = sorted(j, key=lambda x: x[-1])
		for k,v in ex_.items():

			if v[0][0] not in labeled_set: #find the first unlabeled ex

				idx = v[0][0]
				break

		return idx, c_idx

		
	def get_pred_acc(self, fn_test, label_test, al_tl_fn_train, al_tl_label_train,  pseudo_set, pseudo_label):

		fn_train_pred = []
		label_train_pred = []

		if not pseudo_set:
			# print(active_fn_train.shape)
			# print(transfer_fn_train.shape)
			# print(active_label_train.shape)
			# print(transfer_label_train.shape)

			fn_train_pred = np.array(al_tl_fn_train)
			label_train_pred = np.array(al_tl_label_train)
		else:

			# print(fn_train_pred.shape)
			# print(transfer_fn_train.shape)
			# print(label_train_pred.shape)
			# print(transfer_label_train.shape)

			fn_train_pred = self.m_target_fn[pseudo_set]
			fn_train_pred = np.vstack((fn_train_pred, al_tl_fn_train))

			label_train_pred = np.hstack((pseudo_label, al_tl_label_train))


		self.clf.fit(fn_train_pred, label_train_pred)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc


	def plot_confusion_matrix(self, label_test, fn_test):

		fn_preds = self.clf.predict(fn_test)
		acc = accuracy_score(label_test, fn_preds)

		cm_ = CM(label_test, fn_preds)
		cm = normalize(cm_.astype(np.float), axis=1, norm='l1')

		fig = pl.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		fig.colorbar(cax)
		for x in xrange(len(cm)):
			for y in xrange(len(cm)):
				ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
							horizontalalignment='center',
							verticalalignment='center',
							fontsize=10)
		cm_cls =np.unique(np.hstack((label_test,fn_preds)))

		cls = []
		for c in cm_cls:
			cls.append(mapping[c])
		pl.yticks(range(len(cls)), cls)
		pl.ylabel('True label')
		pl.xticks(range(len(cls)), cls)
		pl.xlabel('Predicted label')
		pl.title('Mn Confusion matrix (%.3f)'%acc)

		pl.show()
	
	def get_base_learners(self):
		rf = RFC(n_estimators=100, criterion='entropy')
		svm = SVC(kernel='rbf', probability=True)
		lr = LR()
		self.bl = [rf, lr, svm] #set of base learners
		for b in self.bl:
			b.fit(self.m_source_fd, self.m_source_label) #train each base classifier

###1. cluster
###2. select an example for a label
###3. whether transfer learning can label
###4. if not 3, active learning can label
	
	def askTransferLearner(self, transferLearnerThreshold, class_, nb_c_transfer, nb_f_transfer, idx, p_idx, p_label):

		predLabel = self.bl[0].predict(self.m_target_fd[idx].reshape(1, -1))

		# predProb = self.bl[0].predict_proba(self.m_target_fd[idx].reshape(1, -1)) 
		# predLabel = class_[np.argmax(predProb)]

		return predLabel

		# predMap = {}
		# for wi, b in zip(w, self.bl):
		# 	pr = b.predict_proba(self.m_target_fd[idx].reshape(1, -1))
		# 	predLabel = class_[np.argmax(pr)]

		# 	if predLabel not in predMap.keys():
		# 		predMap.setdefault(predLabel, 0)

		# 	predMap[predLabel] += 1.0
	
		# if idx in p_idx:
		# 	labelIndex = p_idx.index(idx)
		# 	propagatedLabel = p_label[labelIndex]

		# 	if propagatedLabel not in predMap.keys():
		# 		predMap.setdefault(propagatedLabel, 0)
		# 	predMap[propagatedLabel] += 1.0

		# predList = sorted(predMap, key=predMap.__getitem__, reverse=True)

		# pred = predList[0]

		# if idx in p_idx:
		# 	if pred != propagatedLabel:
		# 		print("not propagatedLabel in transfer learning")

		# pred = 0
		# w = []
		# v_c = set(nb_c_transfer[idx])
		# for n in nb_f_transfer:
		# 	v_f = set(n[idx])
		# 	cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based weight
		# 	inter = v_c & v_f
		# 	union = v_c | v_f
		# 	d_i = 0
		# 	d_u = 0
		# 	sim = 0.0
		# 	for it in inter:
		# 		d_i += np.linalg.norm(self.m_target_fn[idx]-self.m_target_fn[it])
		# 	for u in union:
		# 		d_u += np.linalg.norm(self.m_target_fn[idx]-self.m_target_fn[u])
		# 	if len(inter) != 0:
		# 		sim = 1 - (d_i/d_u)/cns

		# 	w.append(sim)

		# if np.mean(w) >= transferLearnerThreshold:
		# 	w[:] = [float(j)/sum(w) for j in w]
		# 	pred_pr = np.zeros(len(class_))
		# 	for wi, b in zip(w,self.bl):
		# 		pr = b.predict_proba(self.m_target_fd[idx].reshape(1,-1))
		# 		pred_pr = pred_pr + wi*pr
		# 	pred = class_[np.argmax(pred_pr)]

		# 	return True, pred
		# else:
		# 	return False, pred

	def transferOrNot(self, transferFeatureList, transferLabelList, activeLabelNum, idx):
		labelNumThreshold = 20

		if activeLabelNum < labelNumThreshold:
			predLabel = self.bl[0].predict(self.m_target_fd[idx].reshape(1, -1))
			# if predLabel == self.m_target_label[idx]:
			# 	transferLabelList.append(1.0)
			# 	transferFeatureList.append(self.m_target_fn[idx])
			# else:
			# 	transferLabelList.append(0.0)
			# 	transferFeatureList.append(self.m_target_fn[idx])
			# self.judgeClassifier.fit(np.array(transferFeatureList), np.array(transferLabelList))
			return False, predLabel

		predLabel = self.bl[0].predict(self.m_target_fd[idx].reshape(1, -1))
		transferFlag = self.judgeClassifier.predict(self.m_target_fn[idx].reshape(1, -1))
		if transferFlag == 1:
			return True, predLabel
		else:
			## when false, we could add more feature and label to estimate the judgeClassifier
			return False, predLabel

	def run_CV(self):

		totalInstanceNum = len(self.m_target_label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]
		np.random.shuffle(indexList)

		totalAccList = [[] for i in range(10)]
		for cvIter in range(10):
			self.judgeClassifier = LR()
			print("cvIter...\t",cvIter)
			trainNum = int(totalInstanceNum*0.9)
			train = indexList[:trainNum]
			test = indexList[trainNum:]

			# train = indexList
			# test = indexList

			# for train, test in kf:
			fn_train = self.m_target_fn[train]
			label_train = self.m_target_label[train]
			fd_train = self.m_target_fd[train]

			fn_test = self.m_target_fn[test]
			label_test = self.m_target_label[test]
			fd_test = self.m_target_fd[test]

			###transfer learning

			self.get_base_learners()

			n_class = 32/2
			c_transfer = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
			c_transfer.fit(fn_train)
			dist_transfer = np.sort(c_transfer.transform(fn_train))
			ex_id_transfer = dd(list) #example id for each C
			# for i,j,k in zip(c_transfer.labels_, xrange(len(fn_train)), dist_transfer):
			# 	ex_id_transfer[i].append(int(j))
			for i,j,k in zip(c_transfer.labels_, train, dist_transfer):
				ex_id_transfer[i].append(int(j))

			nb_c_transfer = dd() #nb from clustering results
			for exx in ex_id_transfer.values():
				exx = np.asarray(exx)
				for e in exx:
					nb_c_transfer[e] = exx[exx!=e]

			nb_f_transfer = [dd(), dd(), dd()] #nb from classification results
			for b,n in zip(self.bl, nb_f_transfer):
				preds = b.predict(fd_train)
				ex_transfer = dd(list)
				for i,j in zip(preds, train):
					ex_transfer[i].append(int(j))
				for exx in ex_transfer.values():
					exx = np.asarray(exx)
					for e in exx:
						n[e] = exx[exx!=e]

			class_ = np.unique(self.m_source_label)

			c = KMeans(init='k-means++', n_clusters=28, n_init=10)
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
			ctr = 0
			###only active label count for the comparison
			accList = []

			activeLabelNum = 0
			activeAccList = []

			transferLearnerThreshold = 0.5

			al_tl_label_train = []
			al_tl_fn_train = []

			transferFeatureList = []
			transferLabelList = []
			transferLabelNum = 0
			# transfer_label_train = []
			# transfer_fn_train = [] ###track ids of instances labeled by transfer learning

			active_label_train = []
			active_fn_train = []

			for ee in ex_N:
				activeLabelFlag = False
				c_idx = ee[0] #cluster id
				idx = ex[c_idx][0][0] #id of ex closest to centroid of cluster

				label_idx = 0

				if len(np.unique(transferLabelList)) > 1:
					self.judgeClassifier.fit(np.array(transferFeatureList), np.array(transferLabelList))
				
				transferLabelFlag, label_transfer = self.transferOrNot(transferFeatureList, transferLabelList, activeLabelNum, idx)

				if transferLabelFlag:
					# label_transfer = self.askTransferLearner(transferLearnerThreshold, class_, nb_c_transfer, nb_f_transfer, idx, p_idx, p_label)
					transferLabelNum += 1.0
					activeLabelFlag = False
					label_idx = label_transfer
					al_tl_label_train.append(label_transfer)
					al_tl_fn_train.append(self.m_target_fn[idx])
				# label, conf = self.askTransferLearner(idx)
				else:
					##active learning

					activeLabelNum += 1.0
					activeLabelFlag = True
					label_idx = self.m_target_label[idx]
					if label_transfer == label_idx:
						transferLabelList.append(1.0)
						transferFeatureList.append(self.m_target_fn[idx])
					else:
						transferLabelList.append(0.0)
						transferFeatureList.append(self.m_target_fn[idx])
					al_tl_label_train.append(self.m_target_label[idx])
					al_tl_fn_train.append(self.m_target_fn[idx])
					## transfer learning
					
				km_idx.append(idx)
				ctr+=1

				if ctr<3:
					if len(np.unique(al_tl_label_train)) < 2:
						if activeLabelFlag:
							activeAccList.append(0.0)
							totalAccList[cvIter].append(0.0)
					else:
						acc = self.get_pred_acc(fn_test, label_test, al_tl_fn_train, al_tl_label_train, p_idx, p_label)
						if activeLabelFlag:
							activeAccList.append(acc)
							totalAccList[cvIter].append(acc)

					# if activeLabelFlag:
					# 	activeAccList.append(acc)

					# accList.append(acc)
					continue

				# self.update_tao(km_idx)
				self.update_tao(al_tl_fn_train, al_tl_label_train)

				# label_idx = 0
				# if activeLabelFlag:
				# 	label_idx = self.m_target_label[idx]
				# else:
				# 	label_idx = label

				p_idx, p_label, p_dist = self.update_pseudo_set(idx, label_idx, c_idx, p_idx, p_label, p_dist)

				acc = self.get_pred_acc(fn_test, label_test, al_tl_fn_train, al_tl_label_train, p_idx, p_label)
				if activeLabelFlag:
					activeAccList.append(acc)
					totalAccList[cvIter].append(acc)

				# accList.append(acc)
				# print acc
				# print("acc\t", acc)

			cl_id = [] #track cluster id on each iter
			ex_al = [] #track ex added on each iter
			fn_test = self.m_target_fn[test]
			label_test = self.m_target_label[test]
			# for rr in range(ctr, rounds):
			while activeLabelNum < rounds:
				fn_train_iter = []
				label_train_iter = []

				if not p_idx:
					# fn_train_iter = self.m_target_fn[km_idx]
					# label_train_iter = self.m_target_label[km_idx]

					fn_train_iter = np.array(al_tl_fn_train)
					label_train_iter = np.array(al_tl_label_train)
				else:
					fn_train_iter = self.m_target_fn[p_idx]
					label_train_iter = p_label

					# fn_train_iter = self.m_target_fn[np.hstack((km_idx, p_idx))]
					# label_train_iter = np.hstack((self.m_target_label[km_idx], p_label))

					fn_train_iter = np.vstack((fn_train_iter, np.array(al_tl_fn_train)))
				
					label_train_iter = np.hstack((label_train_iter, al_tl_label_train))

				self.clf.fit(fn_train_iter, label_train_iter)                        
				idx, c_idx, = self.select_example(km_idx)    

				activeLabelFlag = False
				label_idx = 0

				if len(np.unique(transferLabelList)) > 1:
					self.judgeClassifier.fit(np.array(transferFeatureList), np.array(transferLabelList))
				
				transferLabelFlag, label_transfer = self.transferOrNot(transferFeatureList, transferLabelList, activeLabelNum, idx)

				# transferLabelFlag, label_transfer = self.askTransferLearner(transferLearnerThreshold, class_, nb_c_transfer, nb_f_transfer, idx)			

				if not transferLabelFlag:
					##active 
					activeLabelNum += 1.0
					activeLabelFlag = True
					label_idx = self.m_target_label[idx]

					if label_transfer == label_idx:
						transferLabelList.append(1.0)
						transferFeatureList.append(self.m_target_fn[idx])
					else:
						transferLabelList.append(0.0)
						transferFeatureList.append(self.m_target_fn[idx])

					al_tl_label_train.append(self.m_target_label[idx])
					al_tl_fn_train.append(self.m_target_fn[idx])
				else:
					##transfer
					transferLabelNum += 1.0
					activeLabelFlag = False
					label_idx = label_transfer
					al_tl_label_train.append(label_transfer)
					al_tl_fn_train.append(self.m_target_fn[idx])

				km_idx.append(idx)
				cl_id.append(c_idx) #track picked cluster id on each iteration
				# ex_al.append([rr,key,v[0][-2],self.m_target_label[idx],raw_pt[idx]]) #for debugging

				self.update_tao(al_tl_fn_train, al_tl_label_train)
				# self.update_tao(km_idx)
				# label_idx = 0
				# if activeLabelFlag:
				# 	label_idx = self.m_target_label[idx]
				# else:
				# 	label_idx = label

				p_idx, p_label, p_dist = self.update_pseudo_set(idx, label_idx, c_idx, p_idx, p_label, p_dist)
				
				acc = self.get_pred_acc(fn_test, label_test, al_tl_fn_train, al_tl_label_train, p_idx, p_label)
				# self.acc_sum[rr] = (acc)
				if activeLabelFlag:
					activeAccList.append(acc)
					totalAccList[cvIter].append(acc)

				# accList.append(acc)
			print("transferLabelNum\t", transferLabelNum)
		f = open("al_tl_2.txt", "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()
			# print("active acc\t", len(activeAccList), activeAccList)
			# print("========")
			# print("total acc\t", len(accList), accList)
			# print acc
		# print debug
			# print '# of p label', len(p_label)
			# print cl_id
			# if not p_label:
			#     print 'p label acc', 0
			#     p_acc.append(0)
			# else:
			#     print 'p label acc', sum(self.m_target_label[p_idx]==p_label)/float(len(p_label))
			#     p_acc.append(sum(self.m_target_label[p_idx]==p_label)/float(len(p_label)))
			# print '----------------------------------------------------'
			# print '----------------------------------------------------'

		# print 'class count of clf training ex:', ct(label_train)
		# self.acc_sum = [i for i in self.acc_sum if i]
		# print 'average acc:', [np.mean(i) for i in self.acc_sum]
		# print 'average p label acc:', np.mean(p_acc)

		# self.plot_confusion_matrix(label_test, fn_test)


if __name__ == "__main__":
	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}


	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
	target_label = tmp[:,-1]
	print 'class count of true labels of all ex:\n', ct(target_label)

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


	al = transferActiveLearning(fold, rounds, source_fd, source_label, target_fd, target_label, target_fn)

	al.run_CV()

