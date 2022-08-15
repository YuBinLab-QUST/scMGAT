# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:10:21 2021

@author: 哈哈双翼
"""
import numpy as np
from sklearn import metrics
from munkres import Munkres

class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()


        return print(
            " ACC            =%f \n f1_macro       =%f \n precision_macro=%f \n recall_macro   =%f \n f1_micro       =%f \n precision_micro=%f \n recall_micro   =%f \n NMI            =%f \n ADJ_RAND_SCORE =%f" % (
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

#import pandas as pd
#import torch  
#label   = pd.read_csv("label.csv")     
#label   = label.values
#label   = torch.tensor(label)
#label   = torch.squeeze(label)
#label   = np.array(label.detach().numpy(),dtype=np.float32)
#
#label1   = pd.read_csv("intnmf_label1.csv")     
#label1   = label1.values
#label1   = torch.tensor(label1)
#label1   = torch.squeeze(label1)
#label1   = np.array(label1.detach().numpy(),dtype=np.float32)
#
#
#model = clustering_metrics(label, label1)
#
#model.evaluationClusterModelFromLabel()   
#
#
#hidden_emb   = pd.read_csv("umap.csv", index_col=0)       
#hidden_emb  = np.array(hidden_emb)
#
#  
#metrics.normalized_mutual_info_score(label, label_SCAI)  
#metrics.adjusted_rand_score(label, label_SCAI)    
    
    