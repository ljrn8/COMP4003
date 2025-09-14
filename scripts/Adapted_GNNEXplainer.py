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


def to_graph(data, linegraph=True):
	G = nx.from_pandas_edgelist(
		data,
		source="src",
		target="dst",
		edge_attr=["x", "Attack"],
		create_using=nx.MultiGraph(),
	)

	G = G.to_directed()
	g = dgl.from_networkx(G, edge_attrs=["x", "Attack"])
	if linegraph:
		return g.line_graph(shared=True)
	else:
		return g


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



class CustomGNNExplainer(GNNExplainer):

	params = {
		"ts_coef": 0,
		"motif_coef": 0,
		"sparsity_coef": 0,
		"sparsity_threshold": 0,
		"kde_bw": 0.0005,
	}

	epoch_metrics = {
		"temporal smoothness reward": [],
		"motif coherance reward": [],
		"base loss": [],
	}	

	def __init__(self, node_times, motif_groups, **kwargs):
		super().__init__(**kwargs)
		self.params.update(kwargs)
		self.node_times = node_times
		self.motif_groups = motif_groups


	def temporal_smoothness(self, node_mask, threshold=0.5):
		order = torch.argsort(self.node_times)
		ordered_times = self.node_times[order]
		ordered_node_significance = torch.tensor([max(n) for n in node_mask])[order]

		chosen_times = ordered_times[ordered_node_significance > threshold]
		if len(chosen_times) <= 3:
			return 0.0

		values = chosen_times.cpu().numpy()

		# Hartigan’s dip test
		dip, pval = diptest(values)

		# Option 1: return raw dip (higher = more clusterable)
		return float(dip) # [0,0.25]

		# Option 2 (alternative): use 1 - pval as a "clusterability score"
		# return float(1 - pval)

	def motif_coherance(self, node_mask):
		coherance = sum(
			[
				self.params["motif_coef"] * torch.norm(node_mask[g], p=2)
				for g in self.motif_groups
			]
		)
		return coherance / len(self.motif_groups)

	def additional_loss_terms(self, node_mask):
		reg = 0

		ts = self.temporal_smoothness(node_mask)
		self.epoch_metrics["temporal smoothness reward"].append(ts)
		print(f"temporal smoothness (reward): {ts}")
		reg -= ts

		if len(self.motif_groups) > 0:
			mc = self.motif_coherance(node_mask)
			self.epoch_metrics["motif coherance reward"].append(mc)
			print(f"motif coherance (reward): {mc}")
			reg -= mc

		return reg

	def plot_descent(self):
		with plt.style.context("science"):
			for l in self.epoch_metrics.values():
				plt.plot(l)
			plt.legend(self.epoch_metrics.keys())
			plt.show()

	def _loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		base_loss = super()._loss(y_hat, y)
		reg_loss = self.additional_loss_terms(self.node_mask)
		print(f"base_loss: {base_loss}")
		print(f"total loss: {base_loss + reg_loss}\n")
		self.epoch_metrics["base loss"].append(base_loss)
		return base_loss + reg_loss



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
	if attack == "Dos":
		for hub in scanning_star_nodes.tolist():
			lg_nodes = (src == hub).nonzero(as_tuple=True)[0].tolist()
			if lg_nodes:
				star_motifs.append(lg_nodes)

	fan_motifs = []
	if attack == "DDos":
		for sink in fan_nodes.tolist():
			lg_nodes = (dst == sink).nonzero(as_tuple=True)[0].tolist()
			if lg_nodes:
				fan_motifs.append(lg_nodes)

	explainer = Explainer(
		model=model,
		algorithm=CustomGNNExplainer(
			epochs=100,
			node_times=start_times,
			motif_groups=(star_motifs + fan_motifs),
			tv_coef=1.0,
			motif_coef=0.05,
		),
		explanation_type="phenomenon",
		node_mask_type="attributes",
		edge_mask_type=None,
		model_config=ModelConfig(
			mode="multiclass_classification",
			task_level="node",
			return_type="raw",
		),
	)

	x = subG.x[:, :49]
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
with open('../interm/gnne_adapted_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
    
    