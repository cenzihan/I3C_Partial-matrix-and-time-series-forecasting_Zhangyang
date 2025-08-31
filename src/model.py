import torch
import torch.nn as nn
from torchsummary import summary

def _cnn_block(in_channels, out_channels, kernel_size):
    """A standard CNN block with Conv2D, ReLU activation, and Batch Norm."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class _LstmBranch(nn.Module):
    """
    A Conv-LSTM based branch to process sequences of CSI data.
    It first uses a CNN to extract features from each time step,
    then uses an LSTM to learn temporal dependencies.
    """
    def __init__(self, in_channels, height, width, out_features=64):
        super(_LstmBranch, self).__init__()
        
        # 1. Per-timestep CNN feature extractor
        self.cnn_extractor = nn.Sequential(
            _cnn_block(in_channels, 32, kernel_size=(3, 3)),
            nn.AdaptiveAvgPool2d((1, 1)) # Reduce each feature map to a single value
        )
        
        # 2. LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=2, batch_first=True)
        
        # 3. Output layer
        self.fc = nn.Linear(128, out_features)
        
        # Store dimensions for reshaping
        self.out_features = out_features
        self.height = height
        self.width = width

    def forward(self, x):
        # x shape: [N, SeqLen, C, H, W]
        N, SeqLen, C, H, W = x.shape
        
        # --- CNN Feature Extraction ---
        # We need to process each time step independently.
        # Reshape to treat SeqLen as part of the batch dimension for the CNN.
        x_reshaped = x.view(N * SeqLen, C, H, W)
        cnn_features = self.cnn_extractor(x_reshaped) # -> [N*SeqLen, 32, 1, 1]
        cnn_features = cnn_features.view(N, SeqLen, -1) # -> [N, SeqLen, 32]

        # --- LSTM Temporal Modeling ---
        # LSTM processes the sequence of features.
        # We take the output of the last time step.
        lstm_out, _ = self.lstm(cnn_features) # -> [N, SeqLen, 128]
        last_time_step_out = lstm_out[:, -1, :] # -> [N, 128]
        
        # --- Output Generation ---
        # Pass the final LSTM output through a linear layer.
        fc_out = self.fc(last_time_step_out) # -> [N, out_features]
        
        # --- Reshape back to image-like format ---
        # Since the output is a summary of the sequence, we expand it
        # to match the spatial dimensions [H, W] for the fusion stage.
        out = fc_out.unsqueeze(-1).unsqueeze(-1) # -> [N, out_features, 1, 1]
        out = out.expand(-1, -1, self.height, self.width) # -> [N, out_features, H, W]

        return out

class WeightedSum(nn.Module):
    """
    A custom layer to compute a trainable weighted sum of two tensors.
    alpha * tensor1 + (1 - alpha) * tensor2
    """
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tensor1, tensor2):
        constrained_alpha = torch.sigmoid(self.alpha)
        return constrained_alpha * tensor1 + (1.0 - constrained_alpha) * tensor2

class CsiInpaintingNet(nn.Module):
    """
    The complete CSI inpainting and prediction model, rewritten in PyTorch.
    This version handles complex numbers by splitting them into real and imaginary channels.
    """
    def __init__(self, config):
        super(CsiInpaintingNet, self).__init__()
        
        self.csi_shape = config['model']['csi_shape']
        num_rx, num_tx, num_meas, num_subc = self.csi_shape
        branch_type = config['model'].get('branch_type', 'cnn') # Default to cnn if not specified
        
        # --- Branch 1: Sensing/Inpainting Branch (Always CNN) ---
        self.sensing_branch = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=num_rx * len(config['data']['input_tx_indices']) * 2, 
                out_channels=32, 
                kernel_size=(3, 3), 
                stride=(2, 1), # Upsample height to match temporal branch
                padding=(1, 1),
                output_padding=(1, 0)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            _cnn_block(64, 64, kernel_size=(3, 3))
        )
        
        # --- Branch 2 & 3: Temporal and Fusion Branches (Configurable) ---
        self.weighted_sum = WeightedSum()
        
        if branch_type == 'lstm':
            print("Using LSTM branch for temporal layer.")
            H = num_tx * num_meas # Height for the reshaped tensor
            self.temporal_branch = _LstmBranch(in_channels=num_rx * 2, height=H, width=num_subc, out_features=64)
        else: # Default to CNN
            print("Using CNN branch for temporal layer.")
            self.temporal_branch = nn.Sequential(
                _cnn_block(num_rx * 2, 64, kernel_size=(3, 3)),
                _cnn_block(64, 64, kernel_size=(3, 3))
            )

        # Fusion branch is always CNN-based to process the fused features
        self.fusion_branch = nn.Sequential(
            _cnn_block(64, 64, kernel_size=(3, 3)),
            _cnn_block(64, 64, kernel_size=(3, 3))
        )

        # --- Output Layer ---
        self.output_conv = nn.Conv2d(64, num_rx * 2, (1, 1), padding='same')

    def _split_complex(self, x):
        """Splits a complex tensor into real and imaginary channels."""
        # Input shape: [N, C, H, W]
        # Output shape: [N, 2*C, H, W]
        return torch.cat([x.real, x.imag], dim=1)

    def _split_complex_sequence(self, x):
        """Splits a sequence of complex tensors into real and imaginary channels."""
        # Input shape: [N, SeqLen, C, H, W]
        # Output shape: [N, SeqLen, 2*C, H, W]
        return torch.cat([x.real, x.imag], dim=2)

    def forward(self, partial_csi_input, prev_csi_input):
        # 0. Pre-process inputs by splitting complex into real/imag channels
        # prev_csi_input is now a sequence: [N, SeqLen, Rx, Tx, Meas, Subc]
        N, SeqLen, Rx, Tx, Meas, Subc = prev_csi_input.shape
        
        # Split complex for the sequence input
        prev_csi_real_imag_seq = self._split_complex_sequence(prev_csi_input)
        
        # Split complex for the single time step input
        partial_csi_real_imag = self._split_complex(partial_csi_input)

        # 1. Reshape inputs for 2D convolution
        partial_csi_reshaped = partial_csi_real_imag.view(
            N,
            partial_csi_real_imag.size(1),
            -1,
            Subc
        )
        # Reshape sequence input
        prev_csi_reshaped = prev_csi_real_imag_seq.view(
            N,
            SeqLen,
            prev_csi_real_imag_seq.size(2), # 2 * Rx
            -1,
            Subc
        )
        
        # 2. Sensing Branch
        reconstructed_csi = self.sensing_branch(partial_csi_reshaped)

        # 3. Temporal Branch
        temporal_csi_out = self.temporal_branch(prev_csi_reshaped)
        
        # 4. Fusion Branch
        fused_input = self.weighted_sum(reconstructed_csi, temporal_csi_out)
        fusion_out = self.fusion_branch(fused_input)
        
        # 5. Output Layer
        final_output_real_imag = self.output_conv(fusion_out)
        
        # 6. Combine real and imaginary channels back to a complex tensor
        num_rx = self.csi_shape[0]
        real_part, imag_part = torch.chunk(final_output_real_imag, 2, dim=1)
        final_complex_output = torch.complex(real_part, imag_part)
        
        # Reshape back to the original tensor structure: (N, Rx, Tx, Meas, Subcarriers)
        final_output_reshaped = final_complex_output.view(
            final_complex_output.size(0),
            num_rx,
            self.csi_shape[1],
            self.csi_shape[2],
            self.csi_shape[3]
        )
        
        return final_output_reshaped

if __name__ == '__main__':
    # This block is for testing and will be removed in the final version.
    # It will not work as is because torchsummary doesn't support multiple inputs.
    pass
