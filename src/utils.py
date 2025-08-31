import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="CSI Inpainting and Prediction Model Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()

class CsiDataset(Dataset):
    """
    Custom PyTorch Dataset for loading CSI data from an HDF5 file.
    This version supports loading sequences for multi-step LSTM.
    """
    def __init__(self, config):
        self.config = config
        self.data_cfg = config['data']
        self.train_cfg = config['training']
        self.h5_path = self.data_cfg['path']
        
        with h5py.File(self.h5_path, 'r') as f:
            self.csi_data = f['csi'][:]
        
        num_samples = self.train_cfg.get('num_samples_to_use')
        if num_samples:
            self.csi_data = self.csi_data[:num_samples]
            
        self.tx_indices = torch.tensor(self.data_cfg['input_tx_indices'])
        self.sequence_length = self.train_cfg.get('sequence_length', 1) # Default to 1 for backward compatibility

    def __len__(self):
        # We lose sequence_length samples because we need a history for each item
        return len(self.csi_data) - self.sequence_length

    def __getitem__(self, idx):
        """
        Returns a sequence of past CSI data and the current CSI as the target.
        """
        # Get CSI sequence from t-k to t-1
        start_idx = idx
        end_idx = idx + self.sequence_length
        csi_sequence = self.csi_data[start_idx:end_idx]
        
        # The target is the CSI at time t
        csi_t = self.csi_data[end_idx]

        # --- Create Inputs (X) ---
        # Select specified transmitter antennas from the second dimension
        # Use indexing that preserves dimensions
        partial_csi_t = csi_t[:, self.tx_indices.tolist(), :, :]
        
        # --- Create Label (y) ---
        ground_truth_t = csi_t
        
        # Convert numpy arrays to torch tensors
        partial_csi_t = torch.from_numpy(partial_csi_t).to(torch.complex64)
        csi_sequence = torch.from_numpy(csi_sequence).to(torch.complex64)
        ground_truth_t = torch.from_numpy(ground_truth_t).to(torch.complex64)

        return {
            "partial_csi_input": partial_csi_t,
            "prev_csi_input": csi_sequence # This now holds a sequence
        }, ground_truth_t

    def update_tx_indices(self, new_indices):
        """Allows dynamic updating of the transmitter indices used for partial CSI."""
        print(f"\nUpdating input transmitter indices from {self.tx_indices.tolist()} to {new_indices}")
        self.tx_indices = torch.tensor(new_indices)

def prepare_dataset(config):
    """
    Loads and prepares the dataset for training and validation using PyTorch DataLoader.
    """
    print("--- Preparing Dataset ---")
    
    # 1. Create the full dataset instance
    full_dataset = CsiDataset(config)
    
    # 2. Split into training and validation sets
    train_cfg = config['training']
    val_split = train_cfg['validation_split']
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # 3. Create DataLoaders for batching and shuffling
    batch_size = train_cfg['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )

    print(f"Dataset prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    return train_loader, val_loader

def visualize_and_save(inputs, y_true, y_pred, epoch, val_loss, config):
    """
    Visualizes a sample from a batch and saves it to a file.
    This function is enhanced to show magnitude, phase, and error plots.
    """
    save_dir = config['output']['training_vis_dir']
    tx_indices = config['data']['input_tx_indices']
    os.makedirs(save_dir, exist_ok=True)
    
    # Move tensors to CPU and convert to numpy
    partial_csi_input = inputs['partial_csi_input'][0].cpu().numpy()
    true_csi = y_true[0].cpu().numpy()
    pred_csi = y_pred[0].cpu().numpy()
    
    # Create a zero-padded version of the partial input for visualization
    input_vis = np.zeros_like(true_csi, dtype=np.complex64)
    
    # Place the partial input data at the correct TX antenna positions
    for i, tx_idx in enumerate(tx_indices):
        input_vis[:, tx_idx, :, :] = partial_csi_input[:, 0, :, :]

    # --- Calculations ---
    # Magnitude
    mag_true = np.abs(true_csi)
    mag_pred = np.abs(pred_csi)
    mag_input = np.abs(input_vis)
    mag_error = np.abs(mag_true - mag_pred)

    # Phase (use np.angle for phase in radians)
    phase_true = np.angle(true_csi)
    phase_pred = np.angle(pred_csi)
    phase_input = np.angle(input_vis)
    # Correct for phase wrapping issues in error calculation
    phase_error = np.angle(np.exp(1j * (phase_true - phase_pred)))

    # --- Plotting ---
    # We visualize for one antenna pair: Rx=0, Tx=0, Meas_idx=0
    rx, tx, meas = 0, 0, 0
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(f'CSI Reconstruction - Epoch {epoch} (Val Loss: {val_loss:.4f})\n'
                 f'Showing Rx={rx}, Tx={tx}, Meas={meas}', fontsize=18)

    # --- Magnitude Plots ---
    axes[0, 0].plot(mag_input[rx, tx, meas, :])
    axes[0, 0].set_title('Input Magnitude')
    axes[0, 0].grid(True)

    axes[0, 1].plot(mag_true[rx, tx, meas, :])
    axes[0, 1].set_title('Ground Truth Magnitude')
    axes[0, 1].grid(True)

    axes[0, 2].plot(mag_pred[rx, tx, meas, :])
    axes[0, 2].set_title('Predicted Magnitude')
    axes[0, 2].grid(True)
    
    axes[0, 3].plot(mag_error[rx, tx, meas, :], color='red')
    axes[0, 3].set_title('Magnitude Error')
    axes[0, 3].grid(True)

    # --- Phase Plots ---
    axes[1, 0].plot(phase_input[rx, tx, meas, :], '.-')
    axes[1, 0].set_title('Input Phase')
    axes[1, 0].set_ylabel('Radians')
    axes[1, 0].grid(True)

    axes[1, 1].plot(phase_true[rx, tx, meas, :], '.-')
    axes[1, 1].set_title('Ground Truth Phase')
    axes[1, 1].grid(True)

    axes[1, 2].plot(phase_pred[rx, tx, meas, :], '.-')
    axes[1, 2].set_title('Predicted Phase')
    axes[1, 2].grid(True)

    axes[1, 3].plot(phase_error[rx, tx, meas, :], '.-', color='red')
    axes[1, 3].set_title('Phase Error')
    axes[1, 3].grid(True)

    for ax in axes.flat:
        ax.set_xlabel('Subcarrier Index')

    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nSaved enhanced visualization for epoch {epoch} to {save_path}")
