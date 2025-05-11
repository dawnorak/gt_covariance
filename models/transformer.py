import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from typing import List

EPSILON_EIG = 1e-4  # Small value for eigenvalue clipping in CovarianceDecoder
EPSILON_DIAG_NORM = 1e-6 # Small value for diagonal in R_hat normalization in CovarianceDecoder

class GATv2Layer(nn.Module):
    """Graph Attention Network v2 layer."""
    
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super(GATv2Layer, self).__init__()
        
        self.gat = gnn.GATv2Conv(in_channels=in_features, out_channels=out_features, heads=heads, dropout=dropout, concat=concat,
            # edge_dim=None # if there are edge features
        )
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class SpatialEncoder(nn.Module):
    """Spatial encoder using multiple GATv2 layers."""
    
    def __init__(self,
                 in_features: int,
                 hidden_dim: int = 64,
                 embedding_dim: int = 128,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.1):
        super(SpatialEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        current_dim = in_features
        
        # Layers
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            out_dim = embedding_dim if is_last_layer else hidden_dim
            concat = False if is_last_layer else True # No concat for final layer output
            num_output_heads = 1 if is_last_layer else heads # Single head for final embedding

            self.layers.append(GATv2Layer(
                in_features=current_dim,
                out_features=out_dim,
                heads=num_output_heads,
                dropout=dropout,
                concat=concat
            ))
            current_dim = out_dim * num_output_heads if concat else out_dim
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout_p = dropout
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            node_embeddings: Node-level embeddings [num_nodes, embedding_dim]
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.layers[-1](x, edge_index) # Final layer
        x = self.norm(x) # layernorm
        
        return x

class TemporalEncoder(nn.Module):
    """Temporal encoder using Transformer architecture."""
    
    def __init__(self,
                 d_model: int,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(TemporalEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # sequence as first dimension
            activation=F.relu
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) # Add norm
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
    
    def forward(self, x, src_mask=None):
        """
        Args:
            x: Sequence of node embeddings [batch_size, seq_len, num_nodes, d_model]
            
        Returns:
            temporal_embeddings: Transformed sequence [batch_size, seq_len, num_nodes, d_model]
        """
        batch_size, seq_len, num_nodes, d_model = x.size()
        
        # Reshape to [batch_size * num_nodes, seq_len, d_model] for parallel processing of nodes
        x = x.permute(0, 2, 1, 3).contiguous() # [batch_size, num_nodes, seq_len, d_model]
        x = x.view(batch_size * num_nodes, seq_len, d_model)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        
        # Reshape back to [batch_size, num_nodes, seq_len, d_model]
        x = x.view(batch_size, num_nodes, seq_len, d_model)
        # And then to [batch_size, seq_len, num_nodes, d_model]
        x = x.permute(0, 2, 1, 3).contiguous()
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # og: [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose pe to be [1, max_len, d_model] for easier addition if batch_first=True in x
        self.register_buffer('pe', pe.transpose(0,1)) # Shape [1, max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        # x is [batch_size * num_nodes, seq_len, d_model]
        # self.pe is [1, max_len, d_model]
        # We need self.pe[:, :x.size(1)] to be [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CovarianceDecoder(nn.Module):
    """Decoder for predicting covariance matrix using Volatility-Correlation approach."""
    
    def __init__(self,
                 d_model: int, # Input embedding dimension from temporal encoder
                 num_assets: int,
                 hidden_dim: int = 256, # Hidden dim for MLPs
                 dropout: float = 0.1,
                 epsilon_eig: float = EPSILON_EIG, # For eigenvalue clipping
                 epsilon_diag_norm: float = EPSILON_DIAG_NORM # For normalization
                 ):
        super(CovarianceDecoder, self).__init__()
        
        self.num_assets = num_assets
        self.d_model = d_model
        self.epsilon_eig = epsilon_eig
        self.epsilon_diag_norm = epsilon_diag_norm

        # MLP for predicting volatilities (one per asset)
        self.vol_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) 
        )
        
        # MLP for predicting correlation matrix elements (from pairwise features)
        # Input to this MLP will be features of a pair of assets (element-wise product, d_model)
        self.corr_element_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), # Assumes element-wise product of d_model features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Generate tril indices once
        self.tril_indices = torch.tril_indices(
            row=self.num_assets, col=self.num_assets, offset=-1 # Strict lower triangle
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, num_nodes, d_model]
            
        Returns:
            covariance: Predicted covariance matrix [batch_size, num_assets, num_assets]
        """
        batch_size, _, num_nodes, d_model_actual = x.size()
        if num_nodes != self.num_assets:
            raise ValueError(f"num_nodes in input ({num_nodes}) != self.num_assets ({self.num_assets})")
        if d_model_actual != self.d_model:
            raise ValueError(f"d_model in input ({d_model_actual}) != self.d_model ({self.d_model})")

        # Use the last timestep's embeddings for prediction
        node_embeddings = x[:, -1]  # Shape: [batch_size, num_assets, d_model]
        
        # 1. Predict Volatilities

        # log_vols = self.vol_mlp(node_embeddings).squeeze(-1) # [batch_size, num_assets]
        # volatilities = torch.exp(log_vols) # Ensure positivity
        raw_vols = self.vol_mlp(node_embeddings).squeeze(-1)
        volatilities = F.softplus(raw_vols) + self.epsilon_diag_norm

        # 2. Predict Correlation Matrix

        # Create pairwise features for correlation elements, tril_indices need to be on the same device as x
        tril_indices_device = self.tril_indices.to(x.device)
        
        # Efficiently get features for i and j nodes for all lower triangular pairs
        feat_i = node_embeddings[:, tril_indices_device[0], :] # [batch_size, num_tril_elements, d_model]
        feat_j = node_embeddings[:, tril_indices_device[1], :] # [batch_size, num_tril_elements, d_model]
        
        pair_features = feat_i * feat_j # Element-wise product: [batch_size, num_tril_elements, d_model]
        # Alternative: torch.cat((feat_i, feat_j), dim=-1) -> d_model*2 input for corr_element_mlp
        
        # Predict unconstrained correlation-related values for lower triangle
        # Output: [batch_size, num_tril_elements, 1] -> Squeeze -> [batch_size, num_tril_elements]
        tril_elements_raw = self.corr_element_mlp(pair_features).squeeze(-1)
        
        # Apply tanh to constrain to [-1, 1]
        tril_elements_tanh = torch.tanh(tril_elements_raw)
        
        # Assemble the pre-correlation matrix (R_candidate)
        R_candidate = torch.zeros(batch_size, self.num_assets, self.num_assets, device=x.device)
        R_candidate[:, tril_indices_device[0], tril_indices_device[1]] = tril_elements_tanh
        R_candidate[:, tril_indices_device[1], tril_indices_device[0]] = tril_elements_tanh # Symmetric
        
        # Set diagonal to 1
        # R_candidate.diagonal(dim1=-2, dim2=-1)[:] = 1.0 # In-place modification
        eye = torch.eye(self.num_assets, device=x.device).expand(batch_size, -1, -1)
        R_candidate = R_candidate * (1 - eye) + eye # Zero out diagonal then add identity

        # Eigenvalue clipping to ensure PSD for R_candidate
        # Note: torch.linalg.eigh requires symmetric matrix, which R_candidate is.
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(R_candidate)
        except torch._C._LinAlgError as e:
            print(f"torch.linalg.eigh failed: {e}. R_candidate may have NaNs or Infs.")
            # Fallback: return an identity matrix or handle error appropriately
            jitter = self.epsilon_eig * torch.eye(self.num_assets, device=x.device).unsqueeze(0)
            eigenvalues, eigenvectors = torch.linalg.eigh(R_candidate + jitter)

        clipped_eigenvalues = torch.clamp(eigenvalues, min=self.epsilon_eig)
        R_psd = eigenvectors @ torch.diag_embed(clipped_eigenvalues) @ eigenvectors.transpose(-2, -1)
        
        # Re-normalize R_psd to ensure diagonals are 1 (eigenvalue clipping can change them)
        diag_R_psd = torch.diagonal(R_psd, dim1=-2, dim2=-1)
        # Clamp diagonal elements before sqrt to avoid NaNs if any are zero or negative due to precision
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(diag_R_psd, min=self.epsilon_diag_norm)))
        R = D_inv_sqrt @ R_psd @ D_inv_sqrt
        
        # 3. Construct Covariance Matrix: Sigma = S @ R @ S
        S = torch.diag_embed(volatilities)
        covariance = S @ R @ S
        
        return covariance

class SpatioTemporalGraphTransformer(nn.Module):
    """Complete STGT model for covariance forecasting."""
    
    def __init__(self,
                 in_features: int,
                 num_assets: int,
                 window_size: int, # used by PositionalEncoding max_len if dynamic
                 hidden_dim: int = 64,
                 embedding_dim: int = 128,
                 num_gnn_layers: int = 2,
                 num_transformer_layers: int = 4,
                 gnn_heads: int = 4,
                 transformer_heads: int = 8,
                 dropout: float = 0.1,
                 decoder_hidden_dim_multiplier: int = 4 # Multiplier for decoder's hidden_dim
                 ):
        super(SpatioTemporalGraphTransformer, self).__init__()
        
        self.spatial_encoder = SpatialEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_gnn_layers,
            heads=gnn_heads,
            dropout=dropout
        )
        
        self.temporal_encoder = TemporalEncoder(
            d_model=embedding_dim,
            nhead=transformer_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
            # PositionalEncoding's max_len should be >= window_size
        )
        
        self.decoder = CovarianceDecoder(
            d_model=embedding_dim,
            num_assets=num_assets,
            hidden_dim=hidden_dim * decoder_hidden_dim_multiplier,
            dropout=dropout
        )
    
    def forward(self, graph_sequence: List[Data]) -> torch.Tensor:
        """
        Args:
            graph_sequence: List of batched graphs, each representing a timestep
            
        Returns:
            covariance: Predicted covariance matrix
        """
        if not graph_sequence:
            raise ValueError("Input graph_sequence is empty.")

        batch_size = graph_sequence[0].num_graphs
        seq_len = len(graph_sequence)
        # Assuming all graphs in the batch have the same number of nodes for a given asset set
        # num_nodes_per_graph = graph_sequence[0].x.size(0) / batch_size # This is num_assets
        # graph_sequence[0].x is [total_nodes_in_batch, features]
        # We need num_nodes per graph instance in the batch
        example_graph_x_shape = graph_sequence[0].x.shape
        if example_graph_x_shape[0] % batch_size != 0:
            raise ValueError(f"Total nodes {example_graph_x_shape[0]} not divisible by batch_size {batch_size}")
        num_nodes = example_graph_x_shape[0] // batch_size # num_assets

        spatial_embeddings_list = []
        for t in range(seq_len):
            graph_t = graph_sequence[t]
            # x_t will be [total_nodes_in_batch_t, embedding_dim]
            x_t_encoded = self.spatial_encoder(graph_t.x, graph_t.edge_index)
            
            # Reshape to [batch_size, num_nodes, embedding_dim]
            x_t_reshaped = x_t_encoded.view(batch_size, num_nodes, -1)
            spatial_embeddings_list.append(x_t_reshaped)
        
        # stack along sequence dimension
        # spatial_embeddings is [batch_size, seq_len, num_nodes, embedding_dim]
        spatial_embeddings_tensor = torch.stack(spatial_embeddings_list, dim=1)
        
        temporal_embeddings = self.temporal_encoder(spatial_embeddings_tensor)
        
        covariance = self.decoder(temporal_embeddings)
        
        return covariance