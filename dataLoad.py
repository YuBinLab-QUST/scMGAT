
import numpy as np
import pandas as pd

import dgl
import torch
from torch.utils.data import Dataset

#行=细胞 列=基因
def load_data_atac():
    
    atac  = pd.read_csv("atac.csv", index_col=0).T       
    atac  = np.array(atac)
    atac  = torch.FloatTensor(atac)

    atac_n  = pd.read_csv("atac.csv", index_col=0).T       
    
    g_atac     = dgl.knn_graph(atac, int(atac.size()[0]/10), dist='cosine')
    g_atac     = dgl.remove_self_loop(g_atac)
    
    label_atac_t  = pd.read_csv("label.csv")     
    label_atac_t  = label_atac_t.values
    label_atac_t  = torch.tensor(label_atac_t)
    label_atac_t  = torch.squeeze(label_atac_t)
    
    label_atac_n   = pd.read_csv("label.csv")     
    label_atac_n   = label_atac_n.values
    label_atac_n   = torch.tensor(label_atac_n)
    label_atac_n   = torch.squeeze(label_atac_n)
    label_atac_n   = np.array(label_atac_n.detach().numpy(),dtype=np.float32)
    
    return atac, atac_n, g_atac, label_atac_t, label_atac_n

def load_data_rna():
    
    rna  = pd.read_csv("rna.csv", index_col=0).T    
    rna  = np.array(rna)
    rna  = torch.FloatTensor(rna)
    
    rna_n  = pd.read_csv("rna.csv", index_col=0).T       
    
    g_rna     = dgl.knn_graph(rna, int(rna.size()[0]/10), dist='cosine')
    g_rna     = dgl.remove_self_loop(g_rna)
    
    label_rna_t  = pd.read_csv("label.csv")     
    label_rna_t  = label_rna_t.values
    label_rna_t  = torch.tensor(label_rna_t)
    label_rna_t  = torch.squeeze(label_rna_t)
    
    label_rna_n   = pd.read_csv("label.csv")     
    label_rna_n   = label_rna_n.values
    label_rna_n   = torch.tensor(label_rna_n)
    label_rna_n   = torch.squeeze(label_rna_n)
    label_rna_n   = np.array(label_rna_n.detach().numpy(),dtype=np.float32)
    
    return rna, rna_n, g_rna, label_rna_t, label_rna_n

def load_data_omics():
    
    omics  = pd.read_csv("omics32.csv", index_col=0)
    omics  = np.array(omics)
    omics  = torch.FloatTensor(omics)

    omics_n  = pd.read_csv("omics32.csv", index_col=0)  
    
    g_omics     = dgl.knn_graph(omics, int(omics.size()[0]/10), dist='cosine')
    g_omics     = dgl.remove_self_loop(g_omics)
    
    label_t  = pd.read_csv("label.csv")     
    label_t  = label_t.values
    label_t  = torch.tensor(label_t)
    label_t  = torch.squeeze(label_t)
    
    label_n   = pd.read_csv("label.csv")     
    label_n   = label_n.values
    label_n   = torch.tensor(label_n)
    label_n   = torch.squeeze(label_n)
    label_n   = np.array(label_n.detach().numpy(),dtype=np.float32)
    
    return omics, omics_n, g_omics, label_t, label_n


class TabDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x






