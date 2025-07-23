import pandas as pd
import numpy as np
import dgl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx 
from torch_geometric.utils.convert import from_dgl
from sklearn.utils import class_weight 
from torch import nn
import torch as th

def prepare_flows(data):
    if 'L4_SRC_PORT' in data.columns:
        pk_cols = ( 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT')
        for k in pk_cols:
            assert k in data.columns, f'{k} not in columns {data.columns}'
            data[k] = data[k].apply(str)
            
        data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'].astype(str) + ':' + data['L4_SRC_PORT'].astype(str)
        data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'].astype(str) + ':' + data['L4_DST_PORT'].astype(str)
        data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)

    data = data.drop("Label", axis=1)  # for binary only

    categorical = ['TCP_FLAGS','L7_PROTO','PROTOCOL', 
                        'ICMP_IPV4_TYPE', 'FTP_COMMAND_RET_CODE', 'Attack']
    pk = ['IPV4_SRC_ADDR','IPV4_DST_ADDR']
    numerical = [c for c in data.columns if (c not in categorical) and (c not in pk)]

    # # mean impute infinite/nan
    # def _check(df):
    #     for c in numerical:
    #         n = (~np.isfinite(df[c])).sum()
    #         if n > 0:
    #             print(n, c)
                
    # _check(df)
    # df[numerical] = df[numerical].replace([np.inf, -np.inf], np.nan)
    # df[numerical] = df[numerical].fillna(df[numerical].mean())
    # _check(df)

    # paper used labelencoding for all categories
    le = LabelEncoder()
    for c in categorical:
        data[c] = le.fit_transform(data[c])
    
    # and standradization for the rest
    scaler = StandardScaler()
    data[numerical] = scaler.fit_transform(data[numerical])

    attrs = [c for c in data.columns if c not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")]
    data['h'] = data[attrs].values.tolist()
    return data


def to_graph(data: pd.DataFrame, line_graph=False):
    data['h'] = data['h'].apply(lambda x: np.array(x, dtype=np.float32))
    data['Attack'] = data['Attack'].astype(np.int64)
    data = data.rename(columns={'h': 'x'})
    G = nx.from_pandas_edgelist(data, source='IPV4_SRC_ADDR', 
                                target='IPV4_DST_ADDR', 
                                edge_attr=['x', 'Attack'], 
                                create_using=nx.MultiGraph())
    G = G.to_directed()
    g = dgl.from_networkx(G, edge_attrs=['x', 'Attack'])
    if line_graph:
        g = g.line_graph(shared=True)
        
    return g

def get_weighted_criterion(ref='raw/NF-ToN-IoT-v3.csv'):
    classes_df = pd.read_csv(ref, dtype='category', usecols=['Attack'])
    unique_classes = np.array(classes_df['Attack'].unique())

    # weighted cross entropy loss
    class_weights = class_weight.compute_class_weight(
            class_weight= 'balanced',
                classes=unique_classes,
                y=classes_df['Attack'])

    class_weights = th.FloatTensor(class_weights)
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    del classes_df # memory risk
    return unique_classes


def train_one_graph(model, dgl_linegraph, criterion, train=0.8):
    G = dgl_linegraph
    optimizer = th.optim.Adam(model.parameters())
    model.train()
    
    # test/train masks
    size = G.number_of_nodes()
    train_mask = np.zeros(size)
    train_mask[:int(size*train)] = 1
    test_mask = ~np.array(train_mask, dtype=bool)
    
    optimizer.zero_grad()
    labels =  G.ndata['Attack']
    
    G = from_dgl(G)
    pred = model(G.x, G.edge_index)
    loss = criterion(pred[train_mask, :], labels[train_mask])
    test_loss = criterion(pred[test_mask, :], labels[test_mask])
    
    return loss, test_loss


