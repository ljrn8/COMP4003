## 
#   EGraphSAGE model (TODO: cite)
#   Adapted from supplied source codes
#

# from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import category_encoders as ce
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import dgl
from sklearn.utils import class_weight        

CLASSES = 10

class SAGELayer(nn.Module):
    
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        
        # force to output fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        
        # apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # Eq4
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            # Eq5          
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)
    
    
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score }

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
        

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, CLASSES)
        
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
    
    
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()
       
        
def train(G, model, edge_train_mask, edge_valid_mask, epochs=8_000, test_acc=False):
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(G.edata['Attack'].cpu().numpy()),
                                                 G.edata['Attack'].cpu().numpy())

    class_weights = th.FloatTensor(class_weights).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    
    node_features = G.ndata['h']
    edge_features = G.edata['h']
    edge_label = G.edata['Attack']
    # train_mask = G.edata['train_mask']
    
    opt = th.optim.Adam(model.parameters())

    for epoch in range(1, epochs):
        pred = model(G, node_features, edge_features).cuda()
        loss = criterion(pred[edge_train_mask], edge_label[edge_train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print('Training acc:', compute_accuracy(pred[edge_train_mask], edge_label[edge_train_mask]))
            print('Validation acc:', compute_accuracy(pred[edge_valid_mask], edge_label[edge_valid_mask]))

        if epoch == epochs-1 and test_acc:
            test_mask = ~np.array(edge_train_mask + edge_valid_mask)
            print('\nFinal test acc:', compute_accuracy(pred[test_mask], edge_label[test_mask]))

    

class Preprocessing:
    def _prepare_flows(df):
        data = df
        pk_cols = (
            'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT'
        )
        for k in pk_cols:
            assert k in data.columns, f'{k} not in columns {data.columns}'
            data[k] = data[k].apply(str)
            
        data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
        data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
        data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)
    
        data = data.drop("Label", axis=1) 
    
        df = data 
        categorical = ['TCP_FLAGS','L7_PROTO','PROTOCOL', 
                            'ICMP_IPV4_TYPE', 'FTP_COMMAND_RET_CODE', 'Attack']
        pk = ['IPV4_SRC_ADDR','IPV4_DST_ADDR']
        numerical = [c for c in df.columns if (c not in categorical) and (c not in pk)]
    
        # mean impute infinite/nan
        def _check(df):
            for c in numerical:
                n = (~np.isfinite(df[c])).sum()
                if n > 0:
                    print(n, c)
        _check(df)
        df[numerical] = df[numerical].replace([np.inf, -np.inf], np.nan)
        df[numerical] = df[numerical].fillna(df[numerical].mean())
        _check(df)
    
        # paper used labelencoding for all categories
        le = LabelEncoder()
        for c in categorical:
            df[c] = le.fit_transform(df[c])
        
        # and standradization for the rest
        scaler = StandardScaler()
        df[numerical] = scaler.fit_transform(df[numerical])
    
        attrs = [c for c in data.columns if c not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")]
        data['h'] = data[attrs].values.tolist()
        return df
    
    def flowgraph_encode(df):
        df = Preprocessing._prepare_flows(df)
        G = nx.from_pandas_edgelist(
                df, 
                "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','Attack'],
                create_using=nx.MultiGraph())

        G = G.to_directed()
        G = dgl.from_networkx(G,edge_attrs=['h','Attack'])
        # node data is just ones
        G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])
        return G



"""
class _Preprocessing:
    
    def _prepare_pk(NF_dataframe: pd.Dataframe):
        # Prepares primary graph keys from EGrashSAGE, combining source and IP into addr
        
        data = df = NF_dataframe
        pk_cols = (
            'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT'
        )
        for k in pk_cols:
            assert k in data.columns, f'{k} not in columns {data.columns}'
            data[k] = data[k].apply(str)
            
        data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
        data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
        data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)
        return data

        
    def _graph_encode(NF_dataframe: pd.DataFrame):
        data = NF_dataframe
        data = Preprocessing._prepare_pk(data)
        
        G = nx.from_pandas_edgelist(
            data, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','Attack'],
        create_using=nx.MultiGraph())
    
        
        G = G.to_directed()
        G = from_networkx(G,edge_attrs=['h','Attack'] )
        G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])
        G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)
        G.edata['train_mask'] 
        
        G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))
        G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))
        
        return G 
"""

