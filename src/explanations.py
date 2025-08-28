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
from torch_geometric.explain import Explainer, CaptumExplainer, DummyExplainer, GNNExplainer
from torch_geometric.explain.metric import *
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch_geometric.utils import from_dgl
from tqdm import tqdm
from torch_geometric.explain import ModelConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import dgl
import numpy as np
import scienceplots

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

def plot_sparsity_curves(metrics_list, legend=None, s=3):
    # plt.style.use(['science','no-latex'])
    with plt.style.context('science'): 
           
        for metrics in metrics_list:
            plt.plot(metrics['s'], metrics['fid-'])
            plt.scatter(metrics['s'], metrics['fid-'], s=s)
        
        plt.title('Sparsity Vs Fidelity-')
        if legend: plt.legend(legend)
        plt.show()
        
        for metrics in metrics_list:
            plt.plot(metrics['s'], metrics['fid+'])
            plt.scatter(metrics['s'], metrics['fid+'], s=s)
        
        plt.title('Sparsity Vs Fidelity+')
        if legend: plt.legend(legend)
        plt.show()
        
        for metrics in metrics_list:
            plt.plot(metrics['s'], metrics['c'])
            plt.scatter(metrics['s'], metrics['c'], s=s)
        
        plt.title('Sparsity Vs Characterisation Score')
        if legend: plt.legend(legend)
        plt.show()
        
        
def get_masked_prediction(model, sparsity, explanations, windows_size, df):
    l = {'y': [], 'yp': [], 'ypm': [], 'ypmi': []}
    for batch, expl in zip(get_batch(df, batch_size=windows_size), explanations):
        
        # sparsity restriction is computed per-batch
        # due to memory limitations
        soft_mask = expl.node_mask
        flat_mask = soft_mask.flatten()
        k = int(sparsity * flat_mask.numel())
        threshold = torch.topk(flat_mask, k).values[-1]
        mask = soft_mask > threshold
        
        batch_G = to_graph(batch)
        l['y'].append(batch_G.Attack)
        l['yp'].append(model(batch_G.x, batch_G.edge_index).argmax(axis=1))
        l['ypm'].append(model(batch_G.x * mask, batch_G.edge_index).argmax(axis=1))
        l['ypmi'].append(model(batch_G.x * (~mask), batch_G.edge_index).argmax(axis=1))

    y, y_pred, y_mask, y_imask = [np.concat(ll) for ll in l]
    fidm = np.abs(((y_pred == y).astype(float) - (y_mask == y)).astype(float)).mean()
    fidp = np.abs(((y_pred == y).astype(float) - (y_imask == y)).astype(float)).mean()
    return fidp, fidm


def explain_pyG(G, model, explainer, window_size, df):
    explanations = []
    for batch in tqdm(get_batch(df, batch_size=window_size)):
        batched_G = to_graph(batch)
        explanation = explainer(batched_G.x, 
                                batched_G.edge_index, 
                                target=batched_G.Attack)
        
        explanations.append(explanation)
        
    metrics = {'fid+': [], 'fid-': [], 's': [], 'c': [], 'k': []}
    
    for s in tqdm(np.arange(0.1, 1, 0.1)):
        fn, fp = get_masked_prediction(
            s, 
            explanations=explanations)  
          
        metrics['fid+'].append(fp)
        metrics['fid-'].append(fn)
        c = characterization_score(fp, fn) if (fp * fn) != 0 else 0
        metrics['c'].append(c)
        metrics['s'].append(s)
        
    metrics['softmask fidelity'] = fidelity(explainer, explanation)

    y_pred, ym_pred, ymi_pred = masked_prediction(
        explanation.node_mask, model, G, hardmask=False)

    for idx in range(y_pred.max()):
        attack = target_encoder.inverse_transform([idx])[0]
        fp, fn = fidelities(y_pred= y_pred == idx, 
                            y_mask= ym_pred == idx, 
                            y_imask= ymi_pred == idx,
                            y= G.Attack==idx)

        w = (G.Attack==idx).float().mean()
        c = characterization_score(fp, fn, pos_weight=w, neg_weight=1-w) if fp*fn > 0 else 0
        
        metrics[f'softmask fidelity {attack}'] = fp, fn, c