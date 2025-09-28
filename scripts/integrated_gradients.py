import matplotlib.pyplot as plt
import pickle
import torch as th
import torch.nn.functional as F
import torch
import networkx as nx
import pandas as pd
import numpy as np
import dgl
import copy

import torch

from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
)
from torch_geometric.explain.metric import *
from torch_geometric.explain import ModelConfig
import scienceplots
from explanations import *
import torch, networkx as nx, dgl
from sklearn.metrics import classification_report

from config import INTERM_DIR
from explanations import yeild_attack_problems, to_graph, load_BotIoT

import torch
from tqdm import tqdm


def integrated_gradients(x, edge_index, model, steps, baseline):
    device = x.device
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)

    diff = x - baseline
    total_grads = torch.zeros_like(x, device=device)

    for alpha in tqdm(torch.linspace(0.0, 1.0, steps, device=device)):
        x_scaled = baseline + alpha * diff
        x_scaled.requires_grad_(True)

        # forward pass
        out = model(x_scaled, edge_index)
        # sum to get a scalar and backprop
        model.zero_grad()
        torch.autograd.backward(out.sum(), retain_graph=True)
        grads = x_scaled.grad

        total_grads += grads
        x_scaled.grad.zero_()

    # average gradient then scale by input delta
    avg_grads = total_grads / steps
    attributions = diff * avg_grads

    return attributions


# ------
# script
# ------

with open(INTERM_DIR / "Bot_IoT/label_encoders.pkl", "rb") as f:
    le = pickle.load(f)
    attack_encoder = le["Attack"]

train, test = load_BotIoT()
print("loaded data")

print("\nEvaluating ..\n")
metrics = {}
epoch_metrics = {}
for model, binary_flow, attack in yeild_attack_problems(
    test, models_path=INTERM_DIR / "Bot_IoT/models/", label_encoder=attack_encoder
):

    print(f"\n\n======={attack.upper()}=======\n")

    G = to_graph(binary_flow)
    x = G.x[:, :49]

    # check performance
    logits = model(x, G.edge_index)
    probs = torch.sigmoid(logits)
    y_pred = (probs > 0.5).long()
    print(classification_report(G.Attack, y_pred))

    ig_attr = integrated_gradients(
        x=x,
        edge_index=G.edge_index,
        model=model,
        baseline=torch.zeros_like(x),
        steps=100,
    )
    metrics[attack] = ig_attr
    node_mask = (ig_attr - ig_attr.min()) / (ig_attr.max() - ig_attr.min())

    # softmask metrics
    subG_cp = copy.deepcopy(G)
    subG_cp.x = subG_cp.x[:, :49]
    fp, fn, c = evaluate_softmask(model, subG_cp, node_mask)

    metrics[f"{attack} softmask metrics"] = fp, fn, c
    print(f"\tfp: {fp:.3f}")
    print(f"\tfn: {fn:.3f}")
    print(f"\tc: {c:.3f}")

    # sparsity curve
    metrics[f"{attack} sparsity curve"] = evaluate_sparsity_threshholds(
        model, subG_cp, node_mask
    )
    del subG_cp


# save results
with open(INTERM_DIR / "Bot_IoT/ig.pkl", "wb") as f:
    pickle.dump(metrics, f)
