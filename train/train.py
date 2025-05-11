import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EPSILON_PSD_CHOLESKY = 1e-6 # small value for Cholesky decomposition in loss

def train_model(model, train_loader, val_loader, num_epochs=100, patience=10, lr=0.001,
                horizon_length: int = 5, # Pass horizon length for NLL loss
                epsilon_loss_diag: float = EPSILON_PSD_CHOLESKY):
    """Train the STGT model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Negative Log-Likelihood Loss for multivariate Gaussian returns
    def nll_loss_gaussian(pred_cov, target_returns):
        # pred_cov: [batch_size, num_assets, num_assets]
        # target_returns: [batch_size, horizon, num_assets]
        batch_size, H, N = target_returns.shape
        
        # add small jitter to diagonal for Cholesky stability
        jitter = torch.eye(N, device=pred_cov.device) * epsilon_loss_diag
        pred_cov_stable = pred_cov + jitter.unsqueeze(0)

        try:
            L = torch.linalg.cholesky(pred_cov_stable) # [batch_size, N, N]
        except torch._C._LinAlgError as e:
            print(f"Cholesky failed in loss: {e}. Using pseudo-inverse or skipping batch.")
            # pred_cov_stable = pred_cov + (epsilon_loss_diag + 1e-4) * torch.eye(N, device=pred_cov.device).unsqueeze(0)
            # L = torch.linalg.cholesky(pred_cov_stable)
            return torch.tensor(float('inf'), device=pred_cov.device) # skip this batch by returning large loss


        log_det_term = H * 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1) # Sum over N, H * log_det_Sigma

        # Quadratic term: sum_{h=1 to H} [r_h^T * Sigma_inv * r_h]
        # r_h is target_returns[:, h, :] of shape [batch_size, N]
        # Sigma_inv * r_h^T can be computed using cholesky_solve(r_h.unsqueeze(-1), L)
        # Sum over horizon H
        quadratic_term = torch.zeros(batch_size, device=pred_cov.device)
        for h in range(H):
            r_h = target_returns[:, h, :].unsqueeze(-1) # [batch_size, N, 1]
            # Solve L L^T x = r_h  (Sigma x = r_h) -> x = Sigma_inv r_h
            # L z = r_h -> z = L_inv r_h
            # L^T x = z -> x = (L^T)_inv z
            # -> x = (L^T)_inv L_inv r_h = (L L^T)_inv r_h = Sigma_inv r_h
            # Sigma_inv * r_h (column vector)
            # if L has shape [B, N, N] and r_h has shape [B, N, 1]
            # cholesky_solve expects input B of shape [B, N, K]
            solved_x = torch.cholesky_solve(r_h, L) # [batch_size, N, 1]
            
            # r_h^T * Sigma_inv * r_h
            # r_h.transpose(-2,-1) is [batch_size, 1, N]
            quadratic_term_h = r_h.transpose(-2,-1) @ solved_x # [batch_size, 1, 1]
            quadratic_term += quadratic_term_h.squeeze()

        # Constant term 0.5 * H * N * log(2*pi) is ignored for optimization
        loss_per_sample = 0.5 * (log_det_term + quadratic_term)
        return loss_per_sample.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True 
    )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        
        for graph_seq, _, target_ret_batch in train_loader: # target_cov not used in NLL loss directly
            if not graph_seq: continue

            graph_seq = [g.to(device) for g in graph_seq]
            target_ret_batch = target_ret_batch.to(device)
            
            optimizer.zero_grad()
            pred_cov_batch = model(graph_seq)
            
            loss = nll_loss_gaussian(pred_cov_batch, target_ret_batch)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Encountered inf/nan loss in training epoch {epoch+1}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item() * target_ret_batch.size(0)
        
        train_loss_avg = train_loss_sum / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for graph_seq, _, target_ret_batch_val in val_loader:
                if not graph_seq: continue

                graph_seq = [g.to(device) for g in graph_seq]
                target_ret_batch_val = target_ret_batch_val.to(device)
                
                pred_cov_batch_val = model(graph_seq)
                loss = nll_loss_gaussian(pred_cov_batch_val, target_ret_batch_val)

                if torch.isinf(loss) or torch.isnan(loss):
                    print(f"Warning: Encountered inf/nan loss in validation epoch {epoch+1}. Skipping batch.")
                    continue
                
                val_loss_sum += loss.item() * target_ret_batch_val.size(0)
        
        val_loss_avg = val_loss_sum / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        
        scheduler.step(val_loss_avg)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train NLL: {train_loss_avg:.4f}, Val NLL: {val_loss_avg:.4f}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), 'best_model_psd.pt')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in Val NLL.")
                break
    
    model.load_state_dict(torch.load('best_model_psd.pt'))
    return model

def evaluate_model(model, test_loader, horizon_length: int, epsilon_loss_diag: float = EPSILON_PSD_CHOLESKY):
    """Evaluate the trained model using NLL and other metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_nll_sum = 0.0
    all_pred_covs = []
    all_target_covs_from_data = []

    def nll_loss_gaussian_eval(pred_cov, target_returns):
        batch_size, H, N = target_returns.shape
        jitter = torch.eye(N, device=pred_cov.device) * epsilon_loss_diag
        pred_cov_stable = pred_cov + jitter.unsqueeze(0)
        try:
            L = torch.linalg.cholesky(pred_cov_stable)
        except: return torch.tensor(float('nan'), device=pred_cov.device) # if Cholesky fails
        log_det_term = H * 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)
        quadratic_term = torch.zeros(batch_size, device=pred_cov.device)
        for h in range(H):
            r_h = target_returns[:, h, :].unsqueeze(-1)
            solved_x = torch.cholesky_solve(r_h, L)
            quadratic_term_h = r_h.transpose(-2,-1) @ solved_x
            quadratic_term += quadratic_term_h.squeeze()
        loss_per_sample = 0.5 * (log_det_term + quadratic_term)
        return loss_per_sample.mean()

    with torch.no_grad():
        for graph_seq, target_cov_realized, target_ret_batch_test in test_loader:
            if not graph_seq: continue

            graph_seq = [g.to(device) for g in graph_seq]
            target_ret_batch_test = target_ret_batch_test.to(device)
            
            pred_cov_batch_test = model(graph_seq)
            
            loss = nll_loss_gaussian_eval(pred_cov_batch_test, target_ret_batch_test)
            if not torch.isnan(loss):
                 test_nll_sum += loss.item() * target_ret_batch_test.size(0)
            
            all_pred_covs.append(pred_cov_batch_test.cpu().numpy())
            all_target_covs_from_data.append(target_cov_realized.cpu().numpy()) # target_cov_realized from loader
    
    test_nll_avg = test_nll_sum / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    print(f"Test Negative Log-Likelihood: {test_nll_avg:.4f}")
    
    predictions_np = np.concatenate(all_pred_covs, axis=0)
    targets_np = np.concatenate(all_target_covs_from_data, axis=0)
    
    # Frobenius norm of difference between predicted and realized
    frob_norm_diff = np.mean([np.linalg.norm(p - t, 'fro') for p, t in zip(predictions_np, targets_np)])
    print(f"Average Frobenius Norm (Pred vs Realized): {frob_norm_diff:.4f}")
    
    # Eigenvalue difference
    eig_diffs = []
    for p, t in zip(predictions_np, targets_np):
        try:
            p_eig = np.linalg.eigvalsh(p) # eigvalsh for symmetric matrices
            t_eig = np.linalg.eigvalsh(t)
            eig_diffs.append(np.mean(np.abs(np.sort(p_eig) - np.sort(t_eig))))
        except np.linalg.LinAlgError:
            print("Eigenvalue computation failed for a matrix in evaluation.")
            continue
    if eig_diffs:
      print(f"Average Eigenvalue Difference (Pred vs Realized): {np.mean(eig_diffs):.4f}")
    
    return predictions_np, targets_np, test_nll_avg

# visualize_covariance function
def visualize_covariance(true_cov, pred_cov, index=0, vmin=None, vmax=None):
    """Visualize the predicted vs true covariance matrices."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if vmin is None or vmax is None:
        abs_max = max(np.abs(true_cov[index]).max(), np.abs(pred_cov[index]).max())
        vmin_calc, vmax_calc = -abs_max, abs_max
    else:
        vmin_calc, vmax_calc = vmin, vmax

    # True covariance
    sns.heatmap(true_cov[index], ax=axes[0], cmap='coolwarm', vmin=vmin_calc, vmax=vmax_calc, cbar=True, square=True)
    axes[0].set_title(f'True Covariance (Realized, Index {index})')
    
    # Predicted covariance
    sns.heatmap(pred_cov[index], ax=axes[1], cmap='coolwarm', vmin=vmin_calc, vmax=vmax_calc, cbar=True, square=True)
    axes[1].set_title(f'Predicted Covariance (Index {index})')
    
    # Difference
    diff = true_cov[index] - pred_cov[index]
    abs_max_diff = np.abs(diff).max()
    sns.heatmap(diff, ax=axes[2], cmap='coolwarm', vmin=-abs_max_diff, vmax=abs_max_diff, cbar=True, square=True)
    axes[2].set_title(f'Difference (True - Pred, Index {index})')
    
    plt.tight_layout()
    plt.savefig(f'covariance_comparison_idx{index}.png')
    plt.show()