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
import torch.nn.functional as F
from torch_geometric.explain import GNNExplainer
from NIDSExplainer import NIDSExplainer

import torch_geometric
from torch_geometric.explain import (
    Explainer,
    CaptumExplainer,
    DummyExplainer,
    GNNExplainer,
)
from torch_geometric.explain.metric import *
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch_geometric.utils import from_dgl
from tqdm import tqdm
from torch_geometric.explain import ModelConfig
import scienceplots
from explanations import *
import torch, networkx as nx, dgl
from torch_geometric.transforms import LineGraph
from torch_geometric.utils import from_dgl
from diptest import diptest
from sklearn.metrics import classification_report
import os
from pathlib import Path


def motif_attached_linegraph(processed_flows: pd.DataFrame):
    nx_g = nx.from_pandas_edgelist(
        processed_flows,
        source="src",
        target="dst",
        edge_attr=["x", "Attack"],
        create_using=nx.MultiGraph(),
    )

    nx_g = nx_g.to_directed()

    # DGL graph + edge motifs
    dgl_g = dgl.from_networkx(nx_g, edge_attrs=["x", "Attack"])

    src, dst = dgl_g.edges()
    out_deg = dgl_g.out_degrees()
    in_deg = dgl_g.in_degrees()

    scanning_star_nodes = (out_deg > 10).nonzero(as_tuple=True)[0]
    fan_nodes = (in_deg > 10).nonzero(as_tuple=True)[0]

    is_star = torch.isin(src, scanning_star_nodes).to(torch.uint8)
    is_fan = torch.isin(dst, fan_nodes).to(torch.uint8)
    new = torch.vstack([is_star, is_fan])

    print(dgl_g.edata["x"].shape)
    dgl_g.edata["x"] = torch.hstack([dgl_g.edata["x"], new.T])
    print(dgl_g.edata["x"].shape)

    dgl_lg = dgl_g.line_graph(shared=True)
    pyg_lg = from_dgl(dgl_lg)
    return pyg_lg, dgl_g


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

    G, dgl_g = motif_attached_linegraph(binary_flow)

    x = G.x[:, :49]

    # check performance
    logits = model(x, G.edge_index)
    probs = torch.sigmoid(logits)
    y_pred = (probs > 0.5).long()
    print(classification_report(G.Attack, y_pred))

    # retrive motif information
    is_star, is_fan = G.x[:, 51], G.x[:, 52]
    end_times, start_times = G.x[:, 50], G.x[:, 49]
    scanning_star_nodes = is_star.nonzero(as_tuple=True)[0]
    fan_nodes = is_fan.nonzero(as_tuple=True)[0]
    src, dst = dgl_g.edges()

    star_motifs = []
    if attack == "DoS":
        for hub in scanning_star_nodes.tolist():
            lg_nodes = (src == hub).nonzero(as_tuple=True)[0].tolist()
            if lg_nodes:
                star_motifs.append(lg_nodes)

    fan_motifs = []
    if attack == "DDoS":
        for sink in fan_nodes.tolist():
            lg_nodes = (dst == sink).nonzero(as_tuple=True)[0].tolist()
            if lg_nodes:
                fan_motifs.append(lg_nodes)

    # setup explainer
    explainer = Explainer(
        model=model,
        algorithm=NIDSExplainer(
            epochs=100,
            node_times=start_times,
            motif_groups=(star_motifs + fan_motifs),
            model=model,
            edge_index=G.edge_index,
            x=x,
            tv_coef=2.0,
            motif_coef=0.05,
            align_coef=0.05,
        ),
        explanation_type="phenomenon",
        node_mask_type="attributes",  ## !! should use only level node mask? (object)
        edge_mask_type=None,
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # run explanation
    explanation = explainer(
        x=x,
        edge_index=G.edge_index.to(device),
        target=G.Attack,
    )

    metrics[attack] = explanation

    # softmask metrics
    subG_cp = copy.deepcopy(G)
    subG_cp.x = subG_cp.x[:, :49]
    fp, fn, c = evaluate_softmask(model, subG_cp, explanation.node_mask)
    metrics[f"{attack} softmask metrics"] = fp, fn, c
    print(f"\tfp: {fp:.3f}")
    print(f"\tfn: {fn:.3f}")
    print(f"\tc: {c:.3f}")

    # sparsity curve
    metrics[f"{attack} sparsity curve"] = evaluate_sparsity_threshholds(
        model, subG_cp, explanation.node_mask
    )

    del subG_cp
    epoch_metrics[attack] = explainer.algorithm.epoch_metrics

# save results
metrics["epoch metrics"] = explainer.algorithm.epoch_metrics
with open(INTERM_DIR / "Bot_IoT/gnne_adapted.pkl", "wb") as f:
    pickle.dump(metrics, f)
