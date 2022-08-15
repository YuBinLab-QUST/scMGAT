
import dgl     
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from dataLoad import TabDataset
from model import Gat, Autoencoder
from ae import train_model, get_encodings
from loss import CenterLoss, BiTemperedLogisticLoss

from umap import UMAP 
import matplotlib.pyplot as plt


#***************************data load******************************************
rna_n         = pd.read_csv("rna.csv", index_col=0).T       
atac_n        = pd.read_csv("atac.csv", index_col=0).T  
     
label_atac_t  = pd.read_csv("label.csv")     
label_atac_t  = label_atac_t.values
label_atac_t  = torch.tensor(label_atac_t)
label_atac_t  = torch.squeeze(label_atac_t)

label         = pd.read_csv("label.csv")     
label         = label.values
label         = torch.tensor(label)
label         = torch.squeeze(label)
label         = np.array(label.detach().numpy(),dtype=np.float32)

#****************************************canshu********************************
lr      = 1e-2
epochs  = 200
epoch   = 500
cluster = 13

dur2    = []
list1   = []
list2   = []
#***************************AE train*******************************************

nfeatures_rna = rna_n.shape[1]
nfeatures_atac= atac_n.shape[1]

## concat rna and pro
citeseq       = pd.concat([rna_n, atac_n], axis=1)
citeseq.head()
#rna_n = np.log2(rna_n+1)

train, valid  = train_test_split(citeseq.to_numpy(dtype=np.float32), 
                                 test_size=0.1, random_state=0)
train_ds      = TabDataset(train)
valid_ds      = TabDataset(valid)
train_dl      = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl      = DataLoader(valid_ds, batch_size=64, shuffle=False)
x, y          = next(iter(train_dl))
#train.shape, valid.shape
#x.shape, y.shape

model         = Autoencoder(nfeatures_rna=nfeatures_rna, 
                            nfeatures_pro=nfeatures_atac, 
                            hidden_rna=128, 
                            hidden_pro=16, 
                            z_dim=20)
model, losses = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs)
test_ds       = TabDataset(citeseq.to_numpy(dtype=np.float32))
test_dl       = DataLoader(test_ds, batch_size=64, shuffle=False)
encodings     = get_encodings(model, test_dl)

zz = np.array(encodings.detach().numpy(),dtype=np.float32)
aa = pd.DataFrame(zz)
aa.to_csv('encodings.csv')

#***********************************gat train**********************************


g_encodings   = dgl.knn_graph(encodings, 10, dist='cosine')
g_encodings   = dgl.remove_self_loop(g_encodings)
net           = Gat(g_encodings, encodings.size()[1], hidden_dim=16, out_dim=8, 
                    num_heads=16, num_cluster = cluster)
loss_function = BiTemperedLogisticLoss(reduction='mean', t1=0.2, t2=4.0, 
                                       label_smoothing=0.3)
center        = CenterLoss(num_classes=cluster, feat_dim=8)
opt           = torch.optim.Adam(net.parameters(),  lr = 0.01)

for k in range(epoch):
    if k >= 1:
        t0 = time.time()
    opt.zero_grad()
    
    outputs_atac     = net.forward_feature(encodings)
    outputs1         = net.forward(encodings)
    
    loss_bitll_atac  = loss_function(outputs1, label_atac_t)
    loss_center_atac = center(outputs_atac, label_atac_t)
    loss_nsll        = F.nll_loss(outputs1, label_atac_t)
    loss             = loss_bitll_atac+loss_center_atac+loss_nsll
    
    a_atac           = loss.item()
    list1.append(a_atac)
    list2.append(k)  
    
    loss.backward()
    opt.step()
    if k >= 1:
        dur2.append(time.time() - t0)
    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} ".format(k, loss.item(), 
                                                                np.mean(dur2)))


#############################outdata_fenxi#################3###################
model_umap = UMAP(n_neighbors = 30, min_dist = 1, n_components = 2,
                  metric = 'euclidean',random_state=0)
umap1      = model_umap.fit_transform(outputs_atac.detach().numpy())


kmeans=KMeans(n_clusters=13)
hh = kmeans.fit(umap1)
y_label1=kmeans.labels_


plt.figure(figsize=(8,6))
plt.scatter(umap1[:, 0], umap1[:, 1],c=label,s=5, cmap='Spectral')


ARI = round(metrics.adjusted_rand_score(label, y_label1),3)
NMI = round(metrics.normalized_mutual_info_score(label, y_label1),3)


print('ARI:', ARI)
print('NMI:', NMI)


zzz = np.array(outputs_atac.detach().numpy(),dtype=np.float32)
aaa = pd.DataFrame(zzz)
aaa.to_csv('outputs_jieguo.csv')




