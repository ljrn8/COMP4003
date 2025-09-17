import pandas as pd
import numpy as np
import dgl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import h5py
import networkx as nx
from torch_geometric.utils.convert import from_dgl
import dask.dataframe as dd

import copy
import torch
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from sklearn.metrics import classification_report
from explanations import get_batch, to_graph, yield_class_graphs
from torch import nn
from config import INTERM_DIR

def train_epoch(model, train, test, optimizer, loss_f, batch_size=5000):
    model.train()
    total_loss = 0

    for batch in tqdm(get_batch(train, batch_size=batch_size)):
        G = to_graph(batch)
        optimizer.zero_grad()
        G = G.to(device)
        out = model(G.x.to(device), G.edge_index.to(device)).view(-1)
        loss = loss_f(out, G.Attack.float())

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    total_test_loss = 0
    for batch in tqdm(get_batch(test, batch_size=batch_size)):
        G = to_graph(batch)
        with torch.no_grad():
            out = model(G.x.to(device), G.edge_index.to(device)).view(-1)
            total_test_loss += loss_f(out, G.Attack.float()).item()

    return total_loss / np.ceil(len(train) / batch_size), total_test_loss / np.ceil(
        len(test) / batch_size
    )

def eval(model, test):
    G = to_graph(test)
    logits = model(G.x, G.edge_index)
    probs = torch.sigmoid(logits)
    y_pred = (probs > 0.5).long()
    print(classification_report(G.Attack, y_pred))
    return y_pred


# --------
# script
# --------

device = "cpu"
train = pd.read_csv(INTERM_DIR / "Bot_IoT/BoT_train.csv")
test = pd.read_csv(INTERM_DIR / "Bot_IoT/BoT_test.csv")
train = train[[c for c in train.columns if not c.endswith("_metadata")]]
test = test[[c for c in train.columns if not c.endswith("_metadata")]]

attrs = [c for c in test.columns if c not in ("src", "dst", "Attack", "x")]
test['x'] = test[attrs].values.tolist()
train['x'] = train[attrs].values.tolist()

with open(INTERM_DIR / "Bot_IoT/label_encoders.pkl", "rb") as f:
    le = pickle.load(f)

for behaviour in le["Attack"].classes_:
    if behaviour == "Benign":
        continue

    print(behaviour)
    class_idx = le["Attack"].transform([behaviour])

    binary_train = copy.deepcopy(train)
    binary_train["Attack"] = np.array(binary_train["Attack"]) == class_idx
    binary_test = copy.deepcopy(test)
    binary_test["Attack"] = np.array(binary_test["Attack"]) == class_idx

    model = GraphSAGE(
        49,
        hidden_channels=256,
        out_channels=1,
        num_layers=2,
    ).to(device)

    l, tl = [], []
    for epoch in range(15):
        print(f"epoch: {epoch+1}")
        loss, test_loss = train_epoch(
            model,   
            binary_train,
            test,
            loss_f=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        )
        l.append(loss)
        tl.append(test_loss)
        print(loss, test_loss)

    plt.plot(l)
    plt.plot(tl)
    plt.savefig(f'../figures/{behaviour}_training_curve.png')

    y_pred = eval(model, binary_test)
    
    torch.save(model.state_dict(), INTERM_DIR / f"Bot_IoT/models/{behaviour}")
    del binary_train
    del binary_test
