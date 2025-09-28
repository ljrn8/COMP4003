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
from NIDSExplainer import NIDSExplainer

import torch_geometric
from torch_geometric.explain import (
    Explainer,
    DummyExplainer,
)
from torch_geometric.explain.metric import *
from torch_geometric.explain import ModelConfig
import scienceplots
from explanations import *
import torch, networkx as nx, dgl
from torch_geometric.transforms import LineGraph
from torch_geometric.utils import from_dgl
from sklearn.metrics import classification_report
from pathlib import Path

from config import INTERM_DIR
from explanations import yeild_attack_problems, to_graph, load_BotIoT


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

    explainer = Explainer(
        model=model,
        algorithm=DummyExplainer(),
        explanation_type="phenomenon",
        node_mask_type="attributes",
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

    l = f"{attack} sparsity curve"
    print(f'$$$$ { metrics[l]["fid+"] } ')

    del subG_cp

# save results
with open(INTERM_DIR / "Bot_IoT/de.pkl", "wb") as f:
    pickle.dump(metrics, f)
