import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from scikits.statsmodels.tools.tools import ECDF
from scipy import stats
from scipy.optimize import curve_fit
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
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

        self.clf = LinearSVC()
        self.ex_id = dd(list)

    def run_CV(self):

        kf = KFold(len(self.label), n_folds=self.fold, shuffle=True)
        p_acc = [] #pseudo self.label acc
        self.acc_sum = [[] for i in xrange(rounds)] #acc per iter for each fold

        for train, test in kf:

            fn_train = self.fn[train]
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
            for ee in ex_N:

                key = ee[0] #cluster id
                idx = ex[key][0][0] #id of ex closest to centroid of cluster
                km_idx.append(idx)
                ctr+=1

                if ctr<3:
                    continue

                dist_inter = []
                pair = list(itertools.combinations(km_idx,2))
                for p in pair:
                    d = np.linalg.norm(self.fn[p[0]]-self.fn[p[1]])
                    if self.label[p[0]] != self.label[p[1]]:
                        dist_inter.append(d)
                self.tao = self.alpha_ * min(dist_inter)/2

                #exclude exs
                tmp = []
                #re-visit exs removed on previous itr with the new tao
                idx_tmp=[]
                label_tmp=[]
                for i,j in zip(p_idx,p_label):
                    if p_dist[i]<self.tao:
                        idx_tmp.append(i)
                        label_tmp.append(j)
                    else:
                        p_dist.pop(i)
                        tmp.append(i)
                p_idx = idx_tmp
                p_label = label_tmp

                if ctr==3:
                    #make up for p_self.labels for the first 2 itrs
                    #TBD
                    pass

                for e in self.ex_id[key]:

                    if e == idx:
                        continue

                    d = np.linalg.norm(self.fn[e]-self.fn[idx])
                    if d < self.tao:
                        p_dist[e] = d
                        p_idx.append(e)
                        p_label.append(self.label[idx])
                    else:
                        tmp.append(e)

                if not tmp:
                    self.ex_id.pop(key)
                else:
                    self.ex_id[key] = tmp

                fn_test = self.fn[test]
                label_test = self.label[test]
                if not p_idx:
                    fn_train = self.fn[km_idx]
                    label_train = self.label[km_idx]
                else:
                    fn_train = self.fn[np.hstack((km_idx, p_idx))]
                    label_train = np.hstack((self.label[km_idx], p_label))

                self.clf.fit(fn_train, label_train)
                fn_preds = self.clf.predict(fn_test)

                acc = accuracy_score(label_test, fn_preds)
                self.acc_sum[ctr-3].append(acc)


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
                fn_preds = self.clf.predict(fn_test)

                sub_pred = dd(list) #Mn predicted self.labels for each cluster
                for k,v in self.ex_id.items():
                    sub_pred[k] = self.clf.predict(self.fn[v]) #predict self.labels for cluster learning set

                #entropy-based cluster selection
                rank = []
                for k,v in sub_pred.items():
                    count = ct(v).values()
                    count[:] = [i/float(max(count)) for i in count]
                    H = np.sum(-p*math.log(p,2) for p in count if p!=0)
                    rank.append([k,len(v),H])
                rank = sorted(rank, key=lambda x: x[-1], reverse=True)

                if not rank:
                    break
                idx = rank[0][0] #pick the 1st cluster on the rank
                cl_id.append(idx) #track cluster id on each iteration

                cc = idx #id of the cluster picked by H
                c_id = self.ex_id[cc] #examples in the cluster picked
                sub_label = sub_pred[idx] #used when choosing cluster by H
                sub_fn = self.fn[c_id]

                #sub-cluster the cluster
                c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
                c_.fit(sub_fn)
                dist = np.sort(c_.transform(sub_fn))

                ex_ = dd(list)
                for i,j,k,l in zip(c_.labels_, c_id, dist, sub_label):
                    ex_[i].append([j,l,k[0]])
                for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
                    ex_[i] = sorted(j, key=lambda x: x[-1])
                for k,v in ex_.items():

                    if v[0][0] not in km_idx: #find the first unself.labeled ex

                        idx = v[0][0]
                        km_idx.append(idx)

                        #update tao then remove ex<tao
                        dist_inter = []
                        pair = list(itertools.combinations(km_idx,2))
                        for p in pair:
                            d = np.linalg.norm(self.fn[p[0]]-self.fn[p[1]])
                            if self.label[p[0]] != self.label[p[1]]:
                                dist_inter.append(d)
                        self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2

                        tmp = []
                        #re-visit exs removed on previous itr with the new tao
                        idx_tmp=[]
                        label_tmp=[]
                        for i,j in zip(p_idx,p_label):
                            if p_dist[i] < self.tao:
                                idx_tmp.append(i)
                                label_tmp.append(j)
                            else:
                                p_dist.pop(i)
                                tmp.append(i)
                        p_idx = idx_tmp
                        p_label = label_tmp

                        for e in self.ex_id[cc]:
                            if e == idx:
                                continue
                            d = np.linalg.norm(self.fn[e]-self.fn[idx])
                            if d < self.tao:
                                p_dist[e] = d
                                p_idx.append(e)
                                p_label.append(self.label[idx])
                            else:
                                tmp.append(e)

                        if not tmp:
                            self.ex_id.pop(cc)
                        else:
                            self.ex_id[cc] = tmp

                        ex_al.append([rr,cc,v[0][-2],self.label[idx],raw_pt[idx]])
                        break

                if not p_idx:
                    fn_train = self.fn[km_idx]
                    label_train = self.label[km_idx]
                else:
                    fn_train = self.fn[np.hstack((km_idx, p_idx))]
                    label_train = np.hstack((self.label[km_idx], p_label))

                self.clf.fit(fn_train, label_train)
                fn_preds = self.clf.predict(fn_test)
                acc = accuracy_score(label_test, fn_preds)
                self.acc_sum[rr].append(acc)

            print '# of p self.label', len(p_label)
            print cl_id
            if not p_label:
                print 'psudo self.label acc', 0
                p_acc.append(0)
            else:
                print 'psudo self.label acc', sum(self.label[p_idx]==p_label)/float(len(p_label))
                p_acc.append(sum(self.label[p_idx]==p_label)/float(len(p_label)))
            print '----------------------------------------------------'
            print '----------------------------------------------------'

        print 'class count of clf training ex:', ct(label_train)
        print 'average acc:', [np.mean(i) for i in self.acc_sum]
        print 'average p self.label acc:', np.mean(p_acc)

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
        pl.yself.label('True self.label')
        pl.xticks(range(len(cls)), cls)
        pl.xself.label('Predicted self.label')
        pl.title('Mn Confusion matrix (%.3f)'%acc)
        pl.show()

if __name__ == "__main__":

    raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_soda').readlines()]
    tmp = np.genfromtxt('rice_hour_soda', delimiter=',')
    label = tmp[:,-1]
    print 'class count of true self.labels of all ex:\n', ct(label)

    mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

    fn = get_name_features(raw_pt)
    fold = 10
    rounds = 100
    al = active_learning(fold, rounds, fn, label)

    al.run_CV()

