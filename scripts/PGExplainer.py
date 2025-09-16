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
    PGExplainer
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




def load_data():
	test = pd.read_csv("../interm/BoT_test.csv")
	attrs = [
		c
		for c in test.columns
		if c
		not in (
			"src",
			"dst",
			"Attack",
			"x",
			"IPV4_SRC_ADDR_metadata",
			"L4_SRC_PORT_metadata",
			"IPV4_DST_ADDR_metadata",
			"L4_DST_PORT_metadata",
		)
	]
	test["x"] = test[attrs].values.tolist()
	return test


def motif_linegraph(data: pd.DataFrame):

	# 1) Build NX, then RELABEL to 0..N-1 to avoid gaps/off-by-one
	nx_g = nx.from_pandas_edgelist(
		data,
		source="src",
		target="dst",
		edge_attr=["x", "Attack"],
		create_using=nx.DiGraph(),
	)
	# nx_g = nx.convert_node_labels_to_integers(nx_g, ordering='sorted') # ! ?

	# 2) DGL graph + edge motifs (on *edges*)
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

	# dgl_lg = dgl_g.line_graph(shared=True)
	pyg_lg = from_dgl(dgl_g)
	pyg_lg.num_nodes = int(pyg_lg.edge_index.max()) + 1

	return pyg_lg




## -- script

model = GraphSAGE(49, hidden_channels=256, out_channels=5, num_layers=3).to(device)
model.load_state_dict(th.load("../interm/GraphSAGE_BoTIoT.pth"))
model.eval()
print('loaded model')

test = load_data()
print('loaded data')

pyg_lg = motif_linegraph(test)
print(f"Dos stars: {pyg_lg.x[:, 51].sum()}, DDos sinks: {pyg_lg.x[:, 52].sum()}")

metrics = {}
epoch_metrics = {}


### TODO:
# alignement loss -> MSE

print('\nEvaluating ..\n')
for attack, subG in yield_class_graphs(pyg_lg):
	if attack == "Benign":
		continue

	print(attack)

	is_star, is_fan = subG.x[:, 51], subG.x[:, 52]
	end_times, start_times = subG.x[:, 50], subG.x[:, 49]
	scanning_star_nodes = is_star.nonzero(as_tuple=True)[0]
	fan_nodes = is_fan.nonzero(as_tuple=True)[0]
	src, dst = subG.edge_index[0], subG.edge_index[1]

	star_motifs = []
	if attack == "DoS":
		for hub in scanning_star_nodes.tolist():
			lg_nodes = (src == hub).nonzero(as_tuple=True)[0].tolist()
			if lg_nodes:
				star_motifs.append(lg_nodes)

	fan_motifs = []
	# if attack == "DDoS":
	# 	for sink in fan_nodes.tolist():
	# 		lg_nodes = (dst == sink).nonzero(as_tuple=True)[0].tolist()
	# 		if lg_nodes:
	# 			fan_motifs.append(lg_nodes)

	x = subG.x[:, :49]
	
	expl = PGExplainer()
    
	explainer = Explainer(
		model=model,
		algorithm=expl,
		explanation_type="phenomenon",
		node_mask_type="attributes",
		edge_mask_type=None,
		model_config=ModelConfig(
			mode="multiclass_classification",
			task_level="node",
			return_type="raw",
		),
	)

	explanation = explainer(
		x=x,
		edge_index=subG.edge_index.to(device),
		target=subG.Attack,
	)

	metrics[attack] = explanation
	subG_cp = copy.deepcopy(subG)
	subG_cp.x = subG_cp.x[:, :49]

	# softmask metrics
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


metrics['epoch metrics'] = explainer.algorithm.epoch_metrics
with open('../interm/gnne_adapted_embedds_only_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
    
    