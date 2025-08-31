# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import shutil
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import get_args, load_config, prepare_dataset, visualize_and_save
from model import CsiInpaintingNet
from losses import complex_mse

def main():
    """
    Main function to orchestrate the PyTorch model training process.
    """
    # 1. Load Configuration and Setup
    args = get_args()
    config = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear previous results and create output directories
    result_dir = config['output']['training_vis_dir']
    print(f"Clearing previous results in '{result_dir}'...")
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 2. Prepare Datasets
    train_loader, val_loader = prepare_dataset(config)

    # 3. Build the Model
    print("\n--- Building Model ---")
    model = CsiInpaintingNet(config).to(device)

    # 4. Define Optimizer and Loss Function
    print("\n--- Initializing Optimizer and Loss ---")
    optimizer_cfg = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg['learning_rate'])
    criterion = complex_mse

    # 5. Set up TensorBoard
    log_dir = os.path.join(
        config['output']['log_dir'], 
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 6. Training Loop
    print("\n--- Starting Training ---")
    training_cfg = config['training']
    best_val_loss = float('inf')
    
    # --- Early Stopping Setup ---
    early_stopping_cfg = training_cfg.get('early_stopping', {})
    use_early_stopping = early_stopping_cfg.get('enabled', False)
    patience = early_stopping_cfg.get('patience', 10)
    early_stopping_counter = 0
    if use_early_stopping:
        print(f"Early stopping is ENABLED with patience={patience}.")

    # --- Dynamic Selection Setup ---
    use_dynamic_selection = training_cfg.get('dynamic_antenna_selection', False)
    if use_dynamic_selection:
        # For TX antenna selection mode: we have 2 TX antennas
        num_tx_antennas = config['model']['csi_shape'][1]
        antenna_losses = torch.zeros(num_tx_antennas)
        print("Dynamic TX antenna selection is ENABLED.")
    
    for epoch in range(1, training_cfg['epochs'] + 1):
        # --- Print Current TX Indices ---
        current_indices = train_loader.dataset.dataset.tx_indices.tolist()
        print(f"\n--- Epoch {epoch}/{training_cfg['epochs']} | Current Input TX Indices: {current_indices} ---")

        # --- Dynamic Selection Logic ---
        # Note: Dynamic selection is currently only compatible with RX antenna selection mode
        # For TX antenna selection mode (current), this feature is disabled
        if use_dynamic_selection and epoch > 1 and epoch % 5 == 0:
            print("Dynamic selection is not supported in TX antenna selection mode.")
            # In future: implement logic for switching between TX antennas [0] <-> [1]

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, y_true in train_pbar:
            # Move data to device
            # Note: prev_csi is now a sequence [N, SeqLen, ...]
            partial_csi = inputs['partial_csi_input'].to(device)
            prev_csi = inputs['prev_csi_input'].to(device)
            y_true = y_true.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(partial_csi, prev_csi)
            
            # Compute loss
            loss = criterion(y_pred, y_true)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validating")
        
        with torch.no_grad():
            for i, (inputs, y_true) in enumerate(val_pbar):
                # Move data to device
                partial_csi = inputs['partial_csi_input'].to(device)
                prev_csi = inputs['prev_csi_input'].to(device)
                y_true = y_true.to(device)

                # Forward pass
                y_pred = model(partial_csi, prev_csi)
                
                # Compute loss
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                # --- Per-TX-Antenna Loss Calculation for Dynamic Selection ---
                # Note: Currently disabled as we're using TX antenna selection mode
                if use_dynamic_selection:
                    # In TX antenna mode, we would calculate losses per TX antenna
                    # for tx_idx in range(config['model']['csi_shape'][1]):
                    #     tx_loss = criterion(y_pred[:, :, tx_idx], y_true[:, :, tx_idx])
                    #     antenna_losses[tx_idx] += tx_loss.item()
                    pass

                # For the first batch of the first validation epoch, get a sample for visualization
                if i == 0:
                    val_sample_inputs = {'partial_csi_input': partial_csi.cpu()}
                    val_sample_y_true = y_true.cpu()
                    val_sample_y_pred = y_pred.cpu()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # --- Get Alpha from WeightedSum layer for logging ---
        # The parameter is stored as a tensor, get its value.
        # We access it through model.module if using DataParallel
        model_to_log = model.module if isinstance(model, torch.nn.DataParallel) else model
        alpha_param = model_to_log.weighted_sum.alpha.item()
        constrained_alpha = torch.sigmoid(torch.tensor(alpha_param)).item()
        writer.add_scalar('Alpha/constrained', constrained_alpha, epoch)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Alpha = {constrained_alpha:.4f}")

        # --- Save Best Model & Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(config['output']['model_dir'], "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.6f}")
            early_stopping_counter = 0 # Reset counter on improvement
        elif use_early_stopping:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("\nEarly stopping triggered. Terminating training.")
                break # Exit the training loop

        # --- Visualization ---
        if epoch % config['output']['save_freq'] == 0:
            visualize_and_save(
                val_sample_inputs, 
                val_sample_y_true, 
                val_sample_y_pred, 
                epoch,
                avg_val_loss,
                config
            )

    print("\n--- Training Finished ---")
    writer.close()
    
    # Save the final model
    final_model_path = os.path.join(config['output']['model_dir'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()
