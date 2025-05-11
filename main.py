import torch
import numpy as np
from torch_geometric.loader import DataLoader
from data.asset_graph import AssetGraphDataset
from models.transformer import SpatioTemporalGraphTransformer
from train.train import train_model, evaluate_model, visualize_covariance
from data.asset_graph import collate_fn
from data.load_data import load_market_data

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Hyperparams
    window_size = 20
    horizon = 5 # Target horizon for covariance (and NLL loss)
    batch_size = 32
    num_epochs = 10
    patience = 10
    learning_rate = 0.0005
    return_feature_idx = 0 # IMPORTANT: Set this to the index of returns in your feature dimension

    # Load data
    # Data shape: [timesteps, num_assets, num_features]
    # data[:, :, return_feature_idx] should be asset returns.
    market_data_full = load_market_data(num_timesteps=500, num_assets=5, num_features=4)
    
    # Data dimensions
    num_total_timesteps, num_assets, num_features_data = market_data_full.shape
    print(f"Data loaded: {num_total_timesteps} timesteps, {num_assets} assets, {num_features_data} features.")

    # Split data
    train_ratio, val_ratio = 0.7, 0.15
    min_data_len = window_size + horizon 
    
    train_size = int(num_total_timesteps * train_ratio)
    val_size = int(num_total_timesteps * val_ratio)

    if train_size < min_data_len or val_size < min_data_len or \
       (num_total_timesteps - train_size - val_size) < min_data_len:
        print("Warning: Small dataset for chosen window/horizon. Splitting might result in empty datasets.")
        # Adjust num_timesteps in load_market_data or reduce window/horizon for testing
        if num_total_timesteps < 3 * min_data_len:
             raise ValueError("Dataset too small for train/val/test split with current window/horizon.")


    train_data = market_data_full[:train_size]
    val_data = market_data_full[train_size : train_size + val_size]
    test_data = market_data_full[train_size + val_size :]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    # Create datasets
    # Edge method 'correlation' or 'fully_connected'. 'correlation' can be slow for large graphs if dynamic.
    # Static correlation is computed once.
    edge_creation_method = 'fully_connected' # 'correlation' for dynamic, data-driven graphs
    
    train_dataset = AssetGraphDataset(
        train_data, window_size, horizon, return_feature_idx, edge_method=edge_creation_method)
    val_dataset = AssetGraphDataset(
        val_data, window_size, horizon, return_feature_idx, edge_method=edge_creation_method)
    test_dataset = AssetGraphDataset(
        test_data, window_size, horizon, return_feature_idx, edge_method=edge_creation_method)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Model parameters
    # in_features from data, num_assets from data
    hidden_dim_gnn = 64
    embedding_dim_node = 128 # d_model for transformer and decoder input
    num_gnn_layers = 2
    num_transformer_layers = 3
    gnn_heads = 4
    transformer_heads = 4
    model_dropout = 0.1
    
    model = SpatioTemporalGraphTransformer(
        in_features=num_features_data,
        num_assets=num_assets,
        window_size=window_size, 
        hidden_dim=hidden_dim_gnn,
        embedding_dim=embedding_dim_node,
        num_gnn_layers=num_gnn_layers,
        num_transformer_layers=num_transformer_layers,
        gnn_heads=gnn_heads,
        transformer_heads=transformer_heads,
        dropout=model_dropout
    )
    
    print("Model initialized:")
    print(model)
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_model_params / 1e6:.2f} M")

    # Train
    trained_model = train_model(
        model, train_loader, val_loader, 
        num_epochs=num_epochs, patience=patience, lr=learning_rate,
        horizon_length=horizon # Pass horizon for NLL loss
    )
    
    # Evaluate
    print("\nEvaluating on Test Set:")
    predictions, targets, test_loss = evaluate_model(trained_model, test_loader, horizon_length=horizon)
    
    # Visualize
    if predictions.shape[0] > 0 and targets.shape[0] > 0:
        sample_idx_vis = 0
        sample_true_cov_abs_max = np.abs(targets[sample_idx_vis]).max()
        sample_pred_cov_abs_max = np.abs(predictions[sample_idx_vis]).max()
        vis_abs_max = max(sample_true_cov_abs_max, sample_pred_cov_abs_max, 1e-6)

        visualize_covariance(targets, predictions, index=sample_idx_vis, vmin=-vis_abs_max, vmax=vis_abs_max)
        if predictions.shape[0] > 1:
             sample_idx_vis_2 = min(1, predictions.shape[0]-1)
             visualize_covariance(targets, predictions, index=sample_idx_vis_2, vmin=-vis_abs_max, vmax=vis_abs_max)
    else:
        print("Not enough data in predictions/targets to visualize.")

if __name__ == "__main__":
    main()