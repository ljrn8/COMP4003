from config import UQ_dtypes
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import networkx as nx
import dgl


def reduce_dataset(data, ratio=10):
    benign = len(data[data.Attack == 'Benign'])
    target = benign // ratio
    df_reduced = pd.DataFrame()
    for a, group in data.groupby('Attack'):
        if a == 'Benign':
            df_reduced = pd.concat((df_reduced, group))
        else:
            l = len(group)
            group.reset_index()
            print(a, l, len(group.iloc[:target])) 
            df_reduced = pd.concat((df_reduced, group.iloc[:target]))
            
    return data


def prepare_flows(data, dtypes=UQ_dtypes, reduced=False):
    print('loaded')
    
    if reduced:
        data = reduce_dataset(data)
        
    data.drop('Label', axis=1, inplace=True)
    
    # label encoding
    pk = ['IPV4_SRC_ADDR','IPV4_DST_ADDR']
    categorical = [
        c for c, dtype in dtypes.items() 
        if dtype=='category' and dtype not in pk
    ] + ['Attack']

    encoders = {}
    for c in categorical:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].astype(str))
        encoders[c] = le

    #  standradization
    pk = ['IPV4_DST_ADDR', 'IPV4_SRC_ADDR']
    numerical = [c for c in data.columns if (c not in categorical + pk)]
    data[numerical] = data[numerical].astype('float32')
    data[numerical] = data[numerical].replace([np.inf, -np.inf], np.nan)
    data[numerical] = data[numerical].fillna(data[numerical].mean())
    scaler = StandardScaler()
    data[numerical] = scaler.fit_transform(data[numerical])
    
    # pk for graphs
    data['src'] = data['IPV4_SRC_ADDR'].astype(str) + ':' + data['L4_SRC_PORT'].astype(str)
    data['dst'] = data['IPV4_DST_ADDR'].astype(str) + ':' + data['L4_DST_PORT'].astype(str)
    data.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT'], axis=1, inplace=True)

    attrs = [c for c in data.columns if c not in ("src", "dst", "Attack")]
    data['x'] = data[attrs].values.tolist()
    data['x'] = data['x'].apply(lambda x: np.array(x, dtype=np.float32))
    return data
    


def graph_encode(data, linegraph=True):
    G = nx.from_pandas_edgelist(data, source='src', 
                            target='dst', 
                            edge_attr=['x', 'Attack'], 
                            create_using=nx.MultiGraph())

    G = G.to_directed()
    G = dgl.from_networkx(G, edge_attrs=['x', 'Attack'])
    if linegraph:
        G = G.line_graph(shared=True)
        
    print(G.number_of_nodes(), G.number_of_edges())
    return G
    
    
if __name__ == '__main__':
    import argparse, pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--to', type=str, required=True)
    args = parser.parse_args()
    
    data = prepare_flows(args.data)
    G = graph_encode(data, linegraph=True)
    with open(args.to, 'wb') as f:
        pickle.dump(G, f)