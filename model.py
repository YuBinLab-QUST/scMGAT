
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################AE_model###################################
class LinBnDrop(nn.Sequential):    
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        
        super().__init__(*layers)

####################################bianmaqi_model#############################
class Encoder(nn.Module):    
    def __init__(self, nfeatures_rna, nfeatures_pro, hidden_rna, hidden_pro, z_dim):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro

        if nfeatures_rna > 0:
            self.encoder_rna = LinBnDrop(nfeatures_rna, hidden_rna, p=0.1, act=nn.LeakyReLU())

        if nfeatures_pro > 0:
            self.encoder_protein = LinBnDrop(nfeatures_pro, hidden_pro, p=0.1, act=nn.LeakyReLU())

        # make sure hidden_rna and hidden_pro are set correctly
        hidden_rna = 0 if nfeatures_rna == 0 else hidden_rna
        hidden_pro = 0 if nfeatures_pro == 0 else hidden_pro
        
        self.encoder = LinBnDrop(hidden_rna + hidden_pro, z_dim, act=nn.LeakyReLU())

    def forward(self, x):
        if self.nfeatures_rna > 0 and self.nfeatures_pro > 0:
            x_rna = self.encoder_rna(x[:, :self.nfeatures_rna])#每次取一列
            x_pro = self.encoder_protein(x[:, self.nfeatures_rna:])#每次取一列
            x = self.encoder(torch.cat([x_rna, x_pro], 1))#拼接

        elif self.nfeatures_rna > 0 and self.nfeatures_pro == 0:#单组 rna
            x = self.encoder_rna(x)
            x = self.encoder(x)

        elif self.nfeatures_rna == 0 and self.nfeatures_pro > 0:#单组 adt
            x = self.encoder_protein(x)
            x = self.encoder(x)
        
        return x

########################3##3#######jiemaqi_model########################3######
class Decoder(nn.Module):
    def __init__(self, nfeatures_rna, nfeatures_pro, hidden_rna, hidden_pro, z_dim):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro

        # make sure hidden_rna and hidden_pro are set correctly
        hidden_rna = 0 if nfeatures_rna == 0 else hidden_rna
        hidden_pro = 0 if nfeatures_pro == 0 else hidden_pro

        hidden = hidden_rna + hidden_pro

        self.decoder = nn.Sequential(
            LinBnDrop(z_dim, hidden, act=nn.LeakyReLU()),
            LinBnDrop(hidden, nfeatures_rna + nfeatures_pro, bn=False)
            )

    def forward(self, x):
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, nfeatures_rna, nfeatures_pro, hidden_rna, hidden_pro, z_dim):
        super().__init__()
 
        self.encoder = Encoder(nfeatures_rna, nfeatures_pro, hidden_rna, hidden_pro, z_dim)
        self.decoder = Decoder(nfeatures_rna, nfeatures_pro, hidden_rna, hidden_pro, z_dim)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#####################################GAT_model#################################
class GATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim):
        super(GATLayer, self).__init__()
        self.g       = g
        self.fc      = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1,    bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight,      gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)#归一化每一条入边的注意力系数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z # 每个节点的特征
        self.g.apply_edges(self.edge_attention) # 为每一条边获得其注意力系数
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim , num_heads=1, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        #对中间层使用拼接对最后一层使用求平均
        head_out = [attn_head(h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class Gat(nn.Module):
    def __init__(self, g, in_dim, hidden_dim , out_dim, num_heads, num_cluster):
        super(Gat, self).__init__()
        self.layer1 = MultiHeadGATLayer(g , in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, out_dim, 1)
        self.fc2    = nn.Linear(out_dim, num_cluster)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        h = F.normalize(h, p=2, dim=1)
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)

    def forward_feature(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        h = F.normalize(h, p=2, dim=1)
        return h

