import torch
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple

class AssetGraphDataset(torch.utils.data.Dataset):
    """Dataset for loading sequences of asset graphs with features and adjacency matrices."""
    
    def __init__(self, 
                 time_series_data: np.ndarray,  # shape: [timesteps, num_assets, num_features]
                 window_size: int, horizon: int,
                 return_feature_idx: int = 0, # index of the return feature in num_features
                 edge_threshold: float = 0.5, edge_method: str = 'correlation'):
        """
        Args:
            time_series_data: Historical asset features over time
            window_size: Number of timesteps to use as input
            horizon: Number of timesteps to forecast ahead
            return_feature_idx: Index of the feature representing returns, used for target covariance.
            edge_threshold: Threshold for creating edges between assets
            edge_method: Method to create edges ('correlation', 'mutual_info', etc.)
        """
        self.data = time_series_data
        self.window_size = window_size
        self.horizon = horizon
        self.return_feature_idx = return_feature_idx
        self.edge_threshold = edge_threshold
        self.edge_method = edge_method
        
        if self.horizon < 1:
            raise ValueError("Horizon must be at least 1.")
            
        # number of samples
        self.num_samples = len(time_series_data) - window_size - horizon + 1
        if self.num_samples <= 0:
            raise ValueError("Not enough data for the given window_size and horizon. "
                             f"Need len(data) >= {window_size + horizon}. "
                             f"Got len(data) = {len(time_series_data)}")
        
        # Precompute graph structures if static
        if edge_method == 'static':
            self.edge_index = self._create_edges(self.data) 
    
    def _create_edges(self, data_for_edges: np.ndarray) -> torch.Tensor:
        """Create edges between assets based on correlation or other metrics."""
        num_assets = data_for_edges.shape[1] if data_for_edges.ndim > 1 else data_for_edges.shape[0]

        if self.edge_method == 'correlation':
            if data_for_edges.ndim == 3: # full [timesteps, num_assets, num_features] for static graph
                # use returns over time for correlation
                asset_series = data_for_edges[:, :, self.return_feature_idx] # shape: [timesteps, num_assets]
                if asset_series.shape[0] < 2: # need at least 2 timesteps
                    print(f"Warning: Not enough timesteps ({asset_series.shape[0]}) to compute meaningful static correlation. Defaulting to fully_connected.")
                    return self._create_edges(data_for_edges, edge_method='fully_connected') # Fallback
                corr_matrix = np.corrcoef(asset_series, rowvar=False) # rowvar=False: columns are variables (assets)
            elif data_for_edges.ndim == 2: # [num_assets, num_features] for dynamic graph per timestep
                # Correlate assets based on their feature vectors at a given timestep
                # This assumes features are like samples for each asset
                if data_for_edges.shape[1] < 2: # Need at least 2 features
                     print(f"Warning: Not enough features ({data_for_edges.shape[1]}) to compute meaningful dynamic correlation per timestep. Defaulting to fully_connected.")
                     return self._create_edges(data_for_edges, edge_method='fully_connected') # fallback
                corr_matrix = np.corrcoef(data_for_edges) # rows are variables (assets)
            else:
                raise ValueError(f"Unsupported data shape for correlation edge method: {data_for_edges.shape}")

            np.fill_diagonal(corr_matrix, 0) 
            edges = []
            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    if abs(corr_matrix[i, j]) > self.edge_threshold:
                        edges.append([i, j])
                        edges.append([j, i]) # Add symmetric edge
            
            if not edges:
                return torch.empty((2,0), dtype=torch.long)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return edge_index
        
        elif self.edge_method == 'fully_connected':
            edges = []
            for i in range(num_assets):
                for j in range(num_assets):
                    if i != j:
                        edges.append([i, j])
            if not edges: return torch.empty((2,0), dtype=torch.long)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return edge_index
        
        else:
            raise ValueError(f"Unsupported edge method: {self.edge_method}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
        """
        Get a sequence of graphs, the target covariance matrix, and target returns.
        
        Returns:
            graph_sequence: List of PyG Data objects representing asset graphs over time
            target_cov: Target covariance matrix to predict
            target_returns: Target returns over the horizon (for loss calculation)
        """
        start_idx = idx
        window_end_idx = start_idx + self.window_size
        horizon_end_idx = window_end_idx + self.horizon
        
        window_data = self.data[start_idx:window_end_idx]
        
        graph_sequence = []
        for t in range(self.window_size):
            time_step_data = window_data[t] # shape: [num_assets, num_features]
            x = torch.tensor(time_step_data, dtype=torch.float)
            
            if self.edge_method == 'static' and hasattr(self, 'edge_index'):
                edge_index = self.edge_index
            else: # dynamic edges
                edge_index = self._create_edges(time_step_data) 
            
            graph = Data(x=x, edge_index=edge_index)
            graph_sequence.append(graph)
        
        # target returns for covariance calculation and loss
        # shape: [horizon, num_assets, num_features]
        target_period_full_data = self.data[window_end_idx:horizon_end_idx]
        # shape: [horizon, num_assets]
        target_returns_np = target_period_full_data[:, :, self.return_feature_idx]
        
        if self.horizon == 1:
            # for single day horizon, use outer product of returns r*r^T
            r_t = target_returns_np.squeeze(axis=0) # shape [num_assets]
            target_cov_np = np.outer(r_t, r_t)
        else: # horizon > 1
            # For multi-day horizon, use sample covariance
            if target_returns_np.shape[0] < 2:
                 print(f"Warning: Horizon {self.horizon} but only {target_returns_np.shape[0]} "
                       "return observations. Covariance might be unstable.")
            target_cov_np = np.cov(target_returns_np, rowvar=False, ddof=0) # ddof=0 for MLE, bias=True

        target_cov = torch.tensor(target_cov_np, dtype=torch.float)
        target_returns = torch.tensor(target_returns_np, dtype=torch.float)
        
        return graph_sequence, target_cov, target_returns

def collate_fn(batch):
    """Custom collate function for batching sequences of graphs."""
    graph_sequences = [item[0] for item in batch]
    target_covs = torch.stack([item[1] for item in batch])
    target_returns_list = [item[2] for item in batch] # List of [horizon, num_assets] tensors
    
    target_returns_batch = torch.stack(target_returns_list) # [batch_size, horizon, num_assets]

    batched_sequences = []
    if graph_sequences and graph_sequences[0]: # If there are graphs to process
        for t in range(len(graph_sequences[0])): # Iterate over timesteps in sequence
            graphs_t = [seq[t] for seq in graph_sequences if seq] 
            if graphs_t: # If there are graphs for this timestep
                 batched_graph_t = Batch.from_data_list(graphs_t)
                 batched_sequences.append(batched_graph_t)
            # else: handle cases where a timestep might be missing
    
    return batched_sequences, target_covs, target_returns_batch