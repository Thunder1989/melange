"""
a baseline composed of two stages
first stage, run transfer learning to obtain the training labels
second stage, run active learning to obtain the 
"""

import itertools

from sklearn.mixture import DPGMM
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt

def get_name_features(names):

    name = []
    for i in names:

        s = re.findall('(?i)[a-z]{2,}',i)
        name.append(' '.join(s))

    cv = CV(analyzer='char_wb', ngram_range=(3,4))
    fn = cv.fit_transform(name).toarray()

    return fn

def plot_confusion_matrix(test_label, pred):

    mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu',30:'pos',31:'power',32:'ctrl',33:'fan spd',34:'timer'}
    cm_ = CM(test_label, pred)
    cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=Color.YlOrBr)
    fig.colorbar(cax)
    for x in xrange(len(cm)):
        for y in xrange(len(cm)):
            ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=9)
    cm_cls =np.unique(np.hstack((test_label, pred)))
    cls = []
    for c in cm_cls:
        cls.append(mapping[c])
    pl.yticks(range(len(cls)), cls)
    pl.ylabel('True label')
    pl.xticks(range(len(cls)), cls)
    pl.xlabel('Predicted label')
    pl.title('Confusion Matrix (%.3f)'%(ACC(pred, test_label)))
    pl.show()


def output_labels():
    #re-map class label to 0~N
    u, remap = np.unique(np.append(label,pred), return_inverse=True)
    remap = remap[-len(pred):]#output parameters for testing in Java
    f = open('TL_out','w')
    f.writelines(",".join(str(i) for i in l_id))
    f.write('\n')
    f.writelines(",".join(str(l) for l in remap))
    f.write('\n')
    f.close()


class transfer_learning:

    def __init__(self, train_fd, test_fd, train_label, test_label, test_fn, switch=False):

        self.train_fd = train_fd
        self.train_label = train_label

        self.test_fd = test_fd
        self.test_label = test_label

        self.test_fn = test_fn

        self.bl = []

        if switch == True:

            fd_tmp = self.train_fd
            self.train_fd = self.test_fd
            self.test_fd = fd_tmp
            l_tmp = self.train_label
            self.train_label = self.test_label
            self.test_label = l_tmp


    def get_base_learners(self):

        rf = RFC(n_estimators=100, criterion='entropy')
        svm = SVC(kernel='rbf', probability=True)
        lr = LR()
        self.bl = [rf, lr, svm] #set of base learners
        for b in self.bl:
            b.fit(self.train_fd, self.train_label) #train each base classifier


    def run(self):

        '''
        test direct data feature based transfer accuracy on the new building
        '''
        rf = RFC(n_estimators=100, criterion='entropy')
        rf.fit(self.train_fd, self.train_label)
        pred = rf.predict(self.test_fd)
        print 'data feature transfer testing acc:', ACC(pred, self.test_label)
        plot_confusion_matrix(self.test_label, pred)


        '''
        step1: train base models from bldg1
        '''
        self.get_base_learners()


        '''
        step2: TL with name feature on bldg2
        '''
        label = self.test_label
        class_ = np.unique(self.train_label)

        for b in self.bl:
            print b.score(self.test_fd,label)

        n_class = 32/2
        c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
        c.fit(test_fn)
        dist = np.sort(c.transform(test_fn))
        ex_id = DD(list) #example id for each C
        for i,j,k in zip(c.labels_, xrange(len(test_fn)), dist):
            ex_id[i].append(int(j))

        #getting neighors for each ex
        nb_c = DD() #nb from clustering results
        for exx in ex_id.values():
            exx = np.asarray(exx)
            for e in exx:
                nb_c[e] = exx[exx!=e]

        nb_f = [DD(), DD(), DD()] #nb from classification results
        for b,n in zip(self.bl, nb_f):
            preds = b.predict(self.test_fd)
            ex_ = DD(list)
            for i,j in zip(preds, xrange(len(self.test_fd))):
                ex_[i].append(int(j))
            for exx in ex_.values():
                exx = np.asarray(exx)
                for e in exx:
                    n[e] = exx[exx!=e]

        #use base learners' predicitons
        preds = np.array([999 for i in xrange(len(self.test_fd))])
        acc_ = []
        cov_ = []
        for delta in np.linspace(0.1, 0.5, 5):
            print 'running TL with agreement threshold =', delta
            ct = 0
            t = 0
            true = []
            pred = []
            l_id = []
            output = DD()
            for i in xrange(len(test_fn)):
                #getting C v.s. F similiarity
                w = []
                v_c = set(nb_c[i])
                for n in nb_f:
                    v_f = set(n[i])
                    cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based weight
                    inter = v_c & v_f
                    union = v_c | v_f
                    d_i = 0
                    d_u = 0
                    for it in inter:
                        d_i += np.linalg.norm(test_fn[i]-test_fn[it])
                    for u in union:
                        d_u += np.linalg.norm(test_fn[i]-test_fn[u])
                    if len(inter) != 0:
                        sim = 1 - (d_i/d_u)/cns
                        #sim = (d_i/d_u)/cns

                    if i in output:
                        output[i].extend(['%s/%s'%(len(inter), len(union)), 1-sim])
                    else:
                        output[i] = ['%s/%s'%(len(inter), len(union)), 1-sim]
                    w.append(sim)
                output[i].append(np.mean(w))

                if np.mean(w) >= delta:
                    w[:] = [float(j)/sum(w) for j in w]
                    pred_pr = np.zeros(len(class_))
                    for wi, b in zip(w,self.bl):
                        pr = b.predict_proba(self.test_fd[i].reshape(1,-1))
                        pred_pr = pred_pr + wi*pr
                    preds[i] = class_[np.argmax(pred_pr)]
                    true.append(label[i])
                    pred.append(preds[i])
                    ct+=1
                    l_id.append(i)
                    if preds[i]==label[i]:
                        t+=1

            acc_.append(float(t)/ct)
            cov_.append(float(ct)/len(label))

        print 'acc =',acc_,';'
        print 'cov =',cov_,';'

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

        self.clf = SVC()
        self.ex_id = DD(list)

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

        sub_pred = DD(list) #Mn predicted labels for each cluster
        idx = 0

        for k,v in self.ex_id.items():
            sub_pred[k] = self.clf.predict(self.fn[v]) #predict labels for cluster learning set

        #entropy-based cluster selection
        rank = []
        for k,v in sub_pred.items():
            count = CT(v).values()
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
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))

        ex_ = DD(list)
        for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
            ex_[i].append([j,l,k[0]])
        for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k,v in ex_.items():

            if v[0][0] not in labeled_set: #find the first unlabeled ex

                idx = v[0][0]
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

        acc = ACC(label_test, fn_preds)

        return acc

    def plot_confusion_matrix(self, label_test, fn_test):

        fn_preds = self.clf.predict(fn_test)
        acc = ACC(label_test, fn_preds)

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

        kf = KFold(len(self.label), n_folds=self.fold, shuffle=True)
        p_acc = [] #pseudo self.label acc

        for train, test in kf:

            fn_test = self.fn[test]
            label_test = self.label[test]

            fn_train = self.fn[train]
            c = KMeans(init='k-means++', n_clusters=28, n_init=10)
            c.fit(fn_train)
            dist = np.sort(c.transform(fn_train))

            ex = DD(list) #example id, distance to centroid
            self.ex_id = DD(list) #example id for each C
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
            p_dist = DD()
            #first batch of exs: pick centroid of each cluster, and cluster visited based on its size
            ctr = 0
            for ee in ex_N:

                c_idx = ee[0] #cluster id
                idx = ex[c_idx][0][0] #id of ex closest to centroid of cluster
                km_idx.append(idx)
                ctr+=1

                if ctr<3:
                    continue

                self.update_tao(km_idx)

                p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)

                acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
                self.acc_sum[ctr-1].append(acc)


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
