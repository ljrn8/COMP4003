import matplotlib.pyplot as plt
import pickle
import torch as th
import torch.nn.functional as F
import torch
import numpy as np
import copy
import torch
import torch.nn.functional as F
import scienceplots
import torch, networkx as nx, dgl
import torch_geometric
import os 

from torch_geometric.explain.metric import *
from torch_geometric.explain import ModelConfig
from tqdm import tqdm
from config import INTERM_DIR
from explanations import (
    eval, load_BoT_IoT_flows, yield_trained_binary_problems,
    evaluate_softmask, evaluate_sparsity_threshholds
)
from pathlib import Path
from EGraphSAGE_papercode import PyGWrapper, Model

from global_PGExplainer import PGExplainer

from explanations import fidelities

def load_edge_model(path):
    model = PyGWrapper(
        Model(
            ndim_in=49,
            edim=49,
            ndim_out=256,
            classes=1,  
            activation=F.relu,
            dropout=0.2,
        )
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def masked_prediction(model, G, soft_edge_mask):
    return (
        model(G.x, G.edge_index),
        model(G.x * soft_edge_mask.view(-1, 1), G.edge_index),
        model(G.x * (1-soft_edge_mask).view(-1, 1), G.edge_index),
    )

def evaluate_softmask(model, G, soft_edge_mask):
    """ Node mask evaluations by applying softmasks !! (erronous in theory)
    """
    y_pred, ym_pred, ymi_pred = masked_prediction(model, G, soft_edge_mask)
    fp, fn = fidelities(y_pred > 0.5, ym_pred > 0.5, ymi_pred > 0.5, G.Attack)
    c = characterization_score(fp, fn) if (fp * fn) != 0 else 0
    return fp, fn, c


def evaluate_sparsity_threshholds(model, G, softmask, ticks=np.arange(0.1, 1, 0.1)):
    """ Node mask evaluations by applying thresholded hard masks across specified sparsity levels
    """
    sparsity_curve = {'fid+': [], 'fid-': [], 's': [], 'c': [], 'k': []}
    
    for s in tqdm(ticks):
        flat_mask = softmask.flatten()
        k = int(s * flat_mask.numel())
        threshold = torch.topk(flat_mask, k).values[-1]
        new_mask = (softmask >= threshold).float()
        
        y_pred, ym_pred, ymi_pred = masked_prediction(model, G, new_mask)
        fp, fn = fidelities(y_pred > 0.5, ym_pred > 0.5, ymi_pred > 0.5, G.Attack)
        sparsity_curve['fid+'].append(fp)
        sparsity_curve['fid-'].append(fn)
        
        c = characterization_score(fp, fn) if (fp * fn) != 0 else 0
        sparsity_curve['c'].append(c)
        sparsity_curve['s'].append(s)
        sparsity_curve['k'].append(k)
            
    return sparsity_curve



# ------
# script
# ------

train, test, le = load_BoT_IoT_flows()

metrics = {}
epoch_metrics = {}

path = INTERM_DIR / 'Bot_IoT/edge_models'
models = {
        attack: load_edge_model(Path(path) / attack) for attack in os.listdir(path)
}
    
for model, attack, G in yield_trained_binary_problems(
    flows=test, models_ensemble_dict=models, 
    label_encoder=le, node_classification=False
):
    print(f'x.shape= {G.x.shape}')
    print(f'edge_index.shape= {G.edge_index.shape}')

    explainer = PGExplainer(
        model=model,
        emb_dim=256,
        lr=0.01,
        epochs=100,
    )
    
    edge_mask = explainer.explain(G.x, G.edge_index, G.Attack).view(-1)

    print('*********************')
    print(edge_mask.shape)
    print(G.x.shape)

    # softmask metrics
    fp, fn, c = evaluate_softmask(model, G, edge_mask)
    metrics[f"{attack} softmask metrics"] = fp, fn, c
    print(f"\tfp: {fp:.3f}")
    print(f"\tfn: {fn:.3f}")
    print(f"\tc: {c:.3f}")

    # sparsity curve
    metrics[f"{attack} sparsity curve"] = evaluate_sparsity_threshholds(
        model, G, edge_mask
    )

# save results
with open(INTERM_DIR / "Bot_IoT/pge.pkl", "wb") as f:
    pickle.dump(metrics, f)
