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

class active_learning:

	def __init__(self, fold, rounds, fn, label):

		self.fold = fold
		self.rounds = rounds
		self.acc_sum = [[] for i in xrange(self.rounds)] #acc per iter for each fold

		self.fn = fn
		self.label = label

		self.tao = 0
		self.alpha_ = 1

		self.ex_id = dd(list)


	def update_tao(self, labeled_set):

		dist_inter = []
		pair = list(itertools.combinations(labeled_set,2))

		for p in pair:

			d = np.linalg.norm(self.fn[p[0]]-self.fn[p[1]])
			if self.label[p[0]] != self.label[p[1]]:
				dist_inter.append(d)

		try:
			self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2
		except Exception as e:
			self.tao = self.tao


	def update_pseudo_set(self, new_ex_id, cluster_id, p_idx, p_label, p_dist):

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
			d = np.linalg.norm(self.fn[ex]-self.fn[new_ex_id])

			if d < self.tao:
				p_dist[ex] = d
				p_idx.append(ex)
				p_label.append(self.label[new_ex_id])
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
			sub_pred[k] = self.clf.predict(self.fn[v]) #predict labels for cluster learning set

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
		sub_fn = self.fn[c_ex_id]

		#sub-cluster the cluster
		c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10, random_state=3)
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
				c_ex_id.remove(idx) #update the training set by removing selected ex id

				if len(c_ex_id) == 0:
					self.ex_id.pop(c_idx)
				else:
					self.ex_id[c_idx] = c_ex_id
				
				break

		return idx, c_idx

		
	def get_pred_acc(self, fn_test, label_test, labeled_set, pseudo_set, pseudo_label):

		if not pseudo_set:
			fn_train = self.fn[labeled_set]
			label_train = self.label[labeled_set]
		else:
			fn_train = self.fn[np.hstack((labeled_set, pseudo_set))]
			label_train = np.hstack((self.label[labeled_set], pseudo_label))

		self.clf.fit(fn_train, label_train)
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


	def run_CV(self):

		# kf = KFold(len(self.label), n_folds=self.fold, shuffle=True)
		# p_acc = [] #pseudo self.label acc

		# # for cvIter in range(10):
		cvIter = 0
		# # for train, test in kf:
			
		# 	totalInstanceNum = len(self.label)
		# 	print("totalInstanceNum\t", totalInstanceNum)
		# 	indexList = [i for i in range(totalInstanceNum)]
		# 	np.random.shuffle(indexList)
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
		for foldIndex in range(foldNum):
			self.clf = LinearSVC(random_state=3)

			train = []
			for preFoldIndex in range(foldIndex):
				train.extend(foldInstanceList[preFoldIndex])

			test = foldInstanceList[foldIndex]
			for postFoldIndex in range(foldIndex+1, foldNum):
				train.extend(foldInstanceList[postFoldIndex])

			trainNum = int(totalInstanceNum*0.9)
			# train = indexList[:trainNum]
			# test = indexList[trainNum:]
			# print train, test
			# for train, test in kf:

			fn_test = self.fn[test]
			label_test = self.label[test]

			fn_train = self.fn[train]
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
			ctr = 0

			clusterNum = len(ex_N)
			clusterIndex = 0
			while clusterIndex < 3:
			# for ee in ex_N:
				ee = ex_N[clusterIndex]
				clusterIndex += 1
			# for ee in ex_N:
				c_idx = ee[0] #cluster id
				idx = ex[c_idx][0][0] #id of ex closest to centroid of cluster
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

				self.update_tao(km_idx)

				p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)

				acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
				self.acc_sum[ctr-1].append((acc))
				totalAccList[cvIter].append(acc)
				# print acc
				# print("acc\t", acc)

			cl_id = [] #track cluster id on each iter
			ex_al = [] #track ex added on each iter
			fn_test = self.fn[test]
			label_test = self.label[test]
			for rr in range(ctr, rounds):

				if not p_idx:
					fn_train = self.fn[km_idx]
					label_train = self.label[km_idx]
				else:
					fn_train = self.fn[np.hstack((km_idx, p_idx))]
					label_train = np.hstack((self.label[km_idx], p_label))

				self.clf.fit(fn_train, label_train)                        

				idx, c_idx, = self.select_example(km_idx)                
				km_idx.append(idx)
				cl_id.append(c_idx) #track picked cluster id on each iteration
				# ex_al.append([rr,key,v[0][-2],self.label[idx],raw_pt[idx]]) #for debugging

				self.update_tao(km_idx)
				p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)
				
				acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
				self.acc_sum[rr].append((acc))
				totalAccList[cvIter].append(acc)

			cvIter += 1
			# print acc
		# print debug
			# print '# of p label', len(p_label)
			# print cl_id
			# if not p_label:
			#     print 'p label acc', 0
			#     p_acc.append(0)
			# else:
			#     print 'p label acc', sum(self.label[p_idx]==p_label)/float(len(p_label))
			#     p_acc.append(sum(self.label[p_idx]==p_label)/float(len(p_label)))
			# print '----------------------------------------------------'
			# print '----------------------------------------------------'
		f = open("al_2.txt", "w")
		for i in range(10):
			totalAlNum = len(totalAccList[i])
			for j in range(totalAlNum):
				f.write(str(totalAccList[i][j])+"\t")
			f.write("\n")
		f.close()

		# queryMean = []
		# queryVar = []
		# for queryIndex in range(3, rounds):
		# 	queryAccList = []
		# 	for cvIter in range(10):
		# 		queryAccList.append(self.acc_sum[queryIndex][cvIter])

		# 	queryMean.append(np.mean(queryAccList))
		# 	queryVar.append(np.var(queryAccList))
		# print(queryMean)
		# print(queryVar)
		# print 'class count of clf training ex:', ct(label_train)
		# self.acc_sum = [i for i in self.acc_sum if i]
		# print 'average acc:', [(np.mean(i), np.var(i)) for i in self.acc_sum]
		# print 'average p label acc:', np.mean(p_acc)

		# self.plot_confusion_matrix(label_test, fn_test)


if __name__ == "__main__":

	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
	label = tmp[:,-1]
	print 'class count of true labels of all ex:\n', ct(label)

	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

	fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100
	al = active_learning(fold, rounds, fn, label)

	al.run_CV()

