import torch
import torch.nn.functional as F
from torch_geometric.explain import GNNExplainer
from diptest import diptest


class GraphSAGEWithEmbeds(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model  # keep reference to trained model

    def forward(self, x, edge_index, *args, **kwargs):
        return self.model(x, edge_index, *args, **kwargs)

    def get_embs(self, x, edge_index, return_last: bool = True):
        embs = []
        h = x
        for conv in self.model.convs:   # use trained layers
            h = conv(h, edge_index).relu()
            embs.append(h)
        return embs[-1] if return_last else embs
    
    
class NIDSExplainer(GNNExplainer):

    params = {
        "ts_coef": 0,
        "motif_coef": 0,
        "sparsity_coef": 0,
        "sparsity_threshold": 0,
        "align_coef": 1.0,  # weight for embedding alignment
    }

    epoch_metrics = {
        "temporal smoothness reward": [],
        "motif coherance reward": [],
        "embedding align loss": [],
        "base loss": [],
    }    

    def __init__(self, node_times, motif_groups, model, x, edge_index, **kwargs):
        super().__init__(model=model, **kwargs)
        self.params.update(kwargs)
        self.node_times = node_times
        self.motif_groups = motif_groups
        self.x = x
        self.edge_index = edge_index

        # Pre-compute original node embeddings on the *full* graph
        self.model = GraphSAGEWithEmbeds(model)
        with torch.no_grad():
            self.original_embs = self.model.get_embs(x, edge_index)
            # ^ assumes your model has a method to return intermediate embeddings
            # if not, you can modify forward() to return them

    # -----------------------------
    # Existing custom reward terms
    # -----------------------------
    def temporal_smoothness(self, node_mask, threshold=0.5):
        order = torch.argsort(self.node_times)
        ordered_times = self.node_times[order]
        ordered_node_significance = torch.tensor([max(n) for n in node_mask])[order]

        chosen_times = ordered_times[ordered_node_significance > threshold]
        if len(chosen_times) <= 3:
            return 0.0

        values = chosen_times.cpu().numpy()
        dip, pval = diptest(values)
        return float(dip)

    def motif_coherance(self, node_mask):
        coherance = sum(
            [
                self.params["motif_coef"] * torch.norm(node_mask[g], p=2)
                for g in self.motif_groups
            ]
        )
        return coherance / len(self.motif_groups)

    # -----------------------------
    # New embedding alignment loss
    # -----------------------------
    def embedding_alignment(self, masked_x):
        """
        Run forward-pass on masked subgraph and align embeddings with original.
        """
        sub_embs = self.model.get_embs(masked_x, self.edge_index)
        align_loss = F.mse_loss(sub_embs, self.original_embs)
        self.epoch_metrics["embedding align loss"].append(float(align_loss))
        return self.params["align_coef"] * align_loss

    # -----------------------------
    # Combine everything
    # -----------------------------
    def additional_loss_terms(self, node_mask):
        reg = 0

        ts = self.temporal_smoothness(node_mask)
        self.epoch_metrics["temporal smoothness reward"].append(ts)
        reg -= ts

        if len(self.motif_groups) > 0:
            mc = self.motif_coherance(node_mask)
            self.epoch_metrics["motif coherance reward"].append(mc)
            reg -= mc
            print(f"motif coherance:      {mc}")

        masked_x = self.x * (self.node_mask > 0.5)
        if masked_x.sum() > 0:
            align = self.embedding_alignment(masked_x)
            reg += align
            print(f"embeddings alignment: {align}")

        print(f"temporal smoothness:  {ts}")
        return reg

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, 
            #   masked_x=None, masked_edge_index=None, node_idx=None
              ) -> torch.Tensor:
        
        base_loss = super()._loss(y_hat, y)
        reg_loss = self.additional_loss_terms(self.node_mask, )
        self.epoch_metrics["base loss"].append(float(base_loss))
        print(f"base loss  {base_loss}")
        print(f"reg loss:  {reg_loss}\n") 
        return base_loss + reg_loss



"""



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



"""