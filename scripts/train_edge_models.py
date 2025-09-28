import pandas as pd
import numpy as np
import dgl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx
from torch_geometric.utils.convert import from_dgl
import dask.dataframe as dd

import gc
import copy
import torch
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
from explanations import *
from torch import nn
from config import INTERM_DIR

from EGraphSAGE_papercode import Model, PyGWrapper
from sklearn.metrics import precision_recall_fscore_support



def train_epoch(model, G_train, G_test, optimizer, loss_f):
    model.train()
    optimizer.zero_grad()
    out = model(G_train.x, G_train.edge_index).view(-1)
    loss = loss_f(out, G_train.Attack.float())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(G_test.x, G_test.edge_index).view(-1)
        test_loss = loss_f(out, G_test.Attack.float())

        # eval
        y_pred_train =  (torch.sigmoid(model(G_train.x, G_train.edge_index)) > 0.5).long()
        train_metrics = precision_recall_fscore_support(y_pred_train, G_train.Attack, average='binary', zero_division=0)

        y_pred_test = (torch.sigmoid(model(G_test.x, G_test.edge_index)) > 0.5).long()
        test_metrics = precision_recall_fscore_support(y_pred_test, G_test.Attack, average='binary', zero_division=0)
    
    return loss.item(), test_loss.item(), train_metrics, test_metrics


def eval(model, G_test, save_to=None):
    logits = model(G_test.x, G_test.edge_index)
    probs = torch.sigmoid(logits)
    y_pred = (probs > 0.5).long()

    c = classification_report(G_test.Attack, y_pred)
    print(c)
    if save_to:
        with open(save_to, "w") as f:
            f.write(str(c))

    return y_pred


def load_dataset(train_f, test_f, label_encoder_f):
    train = pd.read_csv(train_f)
    test = pd.read_csv(test_f)
    train = train[[c for c in train.columns if not c.endswith("_metadata")]]
    test = test[[c for c in train.columns if not c.endswith("_metadata")]]
    attrs = [c for c in test.columns if c not in ("src", "dst", "Attack", "x")]
    test["x"] = test[attrs].values.tolist()
    train["x"] = train[attrs].values.tolist()
    print("(data loaded)")

    with open(label_encoder_f, "rb") as f:
        le = pickle.load(f)

    return train, test, le["Attack"]


def yield_edge_attack_problems(train_flows, test_flows, label_encoder):
    for behaviour in label_encoder.classes_:
        if behaviour == "Benign":
            continue

        gc.collect()  # force clean pointers
        print(f"binary problem: {behaviour}")
        class_idx = le.transform([behaviour])
        binary_train = copy.deepcopy(train_flows)
        binary_train["Attack"] = np.array(binary_train["Attack"]) == class_idx
        binary_test = copy.deepcopy(test_flows)
        binary_test["Attack"] = np.array(binary_test["Attack"]) == class_idx

        def to_graph(flows):
            G = nx.from_pandas_edgelist(
                flows,
                source="src",
                target="dst",
                edge_attr=["x", "Attack"],
                create_using=nx.MultiGraph(),
            )
            G = G.to_directed()
            g = dgl.from_networkx(G, edge_attrs=["x", "Attack"])
            return from_dgl(g)

        G_train = to_graph(binary_train)
        G_test = to_graph(binary_test)

        yield behaviour, G_train, G_test

    


# --------
# script
# --------

if __name__ == "__main__":

    device = "cpu"
    train, test, le = load_dataset(
        INTERM_DIR / "Bot_IoT/BoT_train.csv",
        INTERM_DIR / "Bot_IoT/BoT_test.csv",
        label_encoder_f=INTERM_DIR / "Bot_IoT/label_encoders.pkl",
    )

    for behaviour, G_train, G_test in yield_edge_attack_problems(train, test, le):
        G_dgl_tr = PyGWrapper.to_dgl_graph(G_train.edge_index, G_train.x)
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

        l, tl = [], []
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(100):
            y = torch.cat([G_train.Attack, G_test.Attack])
            # on_weight = (y == 1).sum() / len(y)
            # criterion = nn.BCEWithLogitsLoss(
                # pos_weight=torch.tensor(on_weight))
                
            from sklearn.utils.class_weight import compute_class_weight
            y_np = y.cpu().numpy()
            class_weights = compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1]))
            
            print(f'class weight: {class_weights}')
            # criterion = nn.BCEWithLogitsLoss()
            
            train_loss, test_loss, train_metrics, test_metrics = train_epoch(
                model,
                G_train,
                G_test,
                loss_f=criterion,
                optimizer=opt,
            )
            l.append(train_loss)
            tl.append(test_loss)
            
            print(f""" 
            epoch: {epoch+1}
                  train_loss={train_loss:.4f} test_loss={test_loss:.4f}
                  test_prec={test_metrics[0]:.4f}   train_prec={train_metrics[0]:.4f}
                  test_rec={test_metrics[1]:.4f}    train_rec={train_metrics[1]:.4f}
            """)


        plt.plot(l)
        plt.plot(tl)
        plt.savefig(f"../figures/training/{behaviour}_edge_training_curve.png")

        y_pred = eval(
            model, G_test, save_to=INTERM_DIR / f"Bot_IoT/edge_models/{behaviour}.txt"
        )
        torch.save(model.state_dict(), INTERM_DIR / f"Bot_IoT/edge_models/{behaviour}")
