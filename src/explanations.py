from tqdm import tqdm 
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import torch as th
import torch.nn.functional as F
import torch
import networkx as nx
import pandas as pd
import numpy as np
import torch_geometric
import dgl
import numpy as np
import scienceplots
import pickle 

from config import INTERM_DIR
from torch_geometric.explain.metric import *
from torch_geometric.utils import from_dgl
from tqdm import tqdm
from torch_geometric.data import Data 

with open(INTERM_DIR / 'label_encoders.pkl', 'rb') as f:
    le = pickle.load(f) 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(data, batch_size=5000):
    all_data = data.copy()
    while len(all_data) > 0:
        if len(all_data) >= batch_size:
            batch = all_data.sample(batch_size)
            all_data = all_data.drop(batch.index)
            yield batch
        else:
            batch = all_data.copy()
            all_data = all_data.drop(batch.index)
            yield batch

def yield_class_graphs(G):
    benign_idx =  le['Attack'].transform(['Benign'])
    for attack in le['Attack'].classes_:
        idx = le['Attack'].transform([attack])

        mask = (G.Attack == idx) | (G.Attack == benign_idx)
        node_idx = mask.nonzero(as_tuple=True)[0]
        edge_index, _ = torch_geometric.utils.subgraph(
            node_idx,
            G.edge_index,
            relabel_nodes=True,
        )
        
        subG = Data(
            x=G.x[node_idx],
            edge_index=edge_index,
            Attack=G.Attack[node_idx]
        )
        
        yield attack, subG

def to_graph(data):
    G = nx.from_pandas_edgelist(data, source='src', 
                                target='dst', 
                                edge_attr=['x', 'Attack'], 
                                create_using=nx.MultiGraph())
    
    G = G.to_directed()
    g = dgl.from_networkx(G, edge_attrs=[ 'x', 'Attack'])
    g = g.line_graph(shared=True)
    return from_dgl(g) 


def masked_prediction(mask, model, G, hardmask=True):
    if not hardmask:
        inv_mask = 1-mask
    else:
        inv_mask = ~mask
        
    y_pred = model(G.x, G.edge_index).argmax(axis=1)
    ym_pred = model(G.x*mask, G.edge_index).argmax(axis=1)
    ymi_pred = model(G.x*inv_mask, G.edge_index).argmax(axis=1)
    return y_pred, ym_pred, ymi_pred


def fidelities(y_pred, y_mask, y_imask, y):
    fn = ((y_pred == y).float() - (y_mask == y).float()).abs().mean()
    fp = ((y_pred == y).float() - (y_imask == y).float()).abs().mean()
    return fp, fn


def evaluate_softmask(model, G, soft_mask):
    (y_pred, ym_pred, ymi_pred), y_true = masked_prediction(
        soft_mask, model, G, hardmask=False), G.Attack
    
    m = y_true != 0 # 0 = benign
    fp, fn = fidelities(y_pred[m], ym_pred[m], ymi_pred[m], y_true[m])
    c = characterization_score(fp, fn) if (fp * fn) != 0 else 0
    return fp, fn, c


def evaluate_pyG_explainer(model, G, explainer):
    metrics = {}
    for attack, subG in yield_class_graphs(G):
        if attack == 'Benign': 
            continue
        print(attack)
        
        # explain
        explanation = explainer(
            x=subG.x.to(device),
            edge_index=subG.edge_index.to(device),
            target=subG.Attack,
        )
        metrics[attack] = explanation
        
        # softmask metrics
        fp, fn, c = evaluate_softmask(
            model, subG, explanation.node_mask)
        metrics[f'{attack} softmask metrics'] = fp, fn, c
        print(f'\tfp: {fp:.3f}')
        print(f'\tfn: {fn:.3f}')
        print(f'\tc: {c:.3f}')
        
        # sparsity curve
        metrics[f'{attack} sparsity curve'] = evaluate_sparsity_threshholds(
            model, subG, explanation.node_mask
        )  
    
    return metrics
   
   
def evaluate_sparsity_threshholds(model, G, softmask, ticks=np.arange(0.1, 1, 0.1)) :
    sparsity_curve = {'fid+': [], 'fid-': [], 's': [], 'c': [], 'k': []}
    for s in tqdm(ticks):
        flat_mask = softmask.flatten()
        k = int(s * flat_mask.numel())
        threshold = torch.topk(flat_mask, k).values[-1]
        new_mask = (softmask >= threshold).float()
        
        (y_pred, ym_pred, ymi_pred), y_true = masked_prediction(
            new_mask.detach().numpy().astype(bool), model, G, hardmask=True), G.Attack
        m = y_true != 0 # 0 = benign
        
        fp, fn = fidelities(y_pred[m], ym_pred[m], ymi_pred[m], y_true[m])
        sparsity_curve['fid+'].append(fp)
        sparsity_curve['fid-'].append(fn)
        
        c = characterization_score(fp, fn) if (fp * fn) != 0 else 0
        sparsity_curve['c'].append(c)
        sparsity_curve['s'].append(s)
        sparsity_curve['k'].append(k)
            
    return sparsity_curve
       