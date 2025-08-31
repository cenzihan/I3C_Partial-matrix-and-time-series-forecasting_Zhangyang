import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

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
    def __init__(self, in_channels, height, width, out_features, lstm_config):
        super(_LstmBranch, self).__init__()
        
        # 1. Per-timestep CNN feature extractor
        self.cnn_extractor = nn.Sequential(
            _cnn_block(in_channels, 32, kernel_size=(3, 3)),
            # OLD BOTTLENECK: nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate the flattened size after the CNN extractor
        # We need a dummy tensor to calculate the output shape of the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, height, width)
            cnn_out_shape = self.cnn_extractor(dummy_input).shape
            self.flattened_size = cnn_out_shape[1] * cnn_out_shape[2] * cnn_out_shape[3]
            
        self.feature_projection = nn.Linear(self.flattened_size, lstm_config['projection_size'])

        # 2. LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_config['projection_size'], 
            hidden_size=lstm_config['hidden_size'], 
            num_layers=lstm_config['num_layers'], 
            batch_first=True,
            # Dropout is only applied if num_layers > 1
            dropout=lstm_config.get('dropout', 0.0) if lstm_config['num_layers'] > 1 else 0.0
        )
        
        # 3. Output layer
        self.fc = nn.Linear(lstm_config['hidden_size'], out_features)
        
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
        cnn_features = self.cnn_extractor(x_reshaped) # -> [N*SeqLen, C_out, H_out, W_out]
        
        # Flatten and project features before feeding to LSTM
        cnn_features_flattened = cnn_features.view(N * SeqLen, -1) # -> [N*SeqLen, flattened_size]
        projected_features = self.feature_projection(cnn_features_flattened) # -> [N*SeqLen, projection_size]
        
        # Reshape back to sequence format for LSTM
        lstm_input = projected_features.view(N, SeqLen, -1) # -> [N, SeqLen, projection_size]

        # --- LSTM Temporal Modeling ---
        # LSTM processes the sequence of features.
        # We take the output of the last time step.
        lstm_out, _ = self.lstm(lstm_input) # -> [N, SeqLen, hidden_size]
        last_time_step_out = lstm_out[:, -1, :] # -> [N, hidden_size]
        
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
    A layer that computes a learnable weighted sum of two input tensors.
    The weight 'alpha' is a learnable parameter.
    """
    def __init__(self, initial_value=0.5, is_trainable=True):
        super(WeightedSum, self).__init__()
        # Initialize alpha as a learnable parameter with a sigmoid function
        # to ensure it stays between 0 and 1. We initialize the raw logit.
        # logit(p) = log(p / (1 - p))
        initial_logit = torch.tensor(np.log(initial_value / (1 - initial_value)), dtype=torch.float32)
        self.alpha_logit = nn.Parameter(initial_logit, requires_grad=is_trainable)

    def forward(self, x1, x2):
        # Apply sigmoid to get alpha in the range [0, 1]
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * x1 + (1 - alpha) * x2

class CsiInpaintingNet(nn.Module):
    """
    The complete CSI inpainting and prediction model, rewritten in PyTorch.
    This version handles complex numbers by splitting them into real and imaginary channels.
    """
    def __init__(self, config):
        super(CsiInpaintingNet, self).__init__()
        self.config = config
        self.sequence_length = config['training']['sequence_length']
        
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
        
        # --- Branch 2: Temporal Branch (Configurable) ---
        branch_type = self.config['model']['branch_type']
        if branch_type == 'lstm':
            print("Using LSTM branch for temporal layer.")
            H = num_tx * num_meas # Height for the reshaped tensor
            # Ensure the 'lstm' config exists
            if 'lstm' not in self.config['model']:
                raise ValueError("LSTM configuration is missing in the model config.")
            lstm_config = self.config['model']['lstm']
            self.temporal_branch = _LstmBranch(
                in_channels=num_rx * 2, 
                height=H, 
                width=num_subc, 
                out_features=64,
                lstm_config=lstm_config
            )
        else: # Default to CNN
            print("Using CNN branch for temporal layer.")
            self.temporal_branch = nn.Sequential(
                _cnn_block(num_rx * 2, 64, kernel_size=(3, 3)),
                _cnn_block(64, 64, kernel_size=(3, 3))
            )

        # -- Weighted Sum Fusion --
        ws_config = self.config['model']['weighted_sum']
        self.weighted_sum = WeightedSum(
            initial_value=ws_config['alpha_initial_value'],
            is_trainable=ws_config['alpha_is_trainable']
        )
        
        # --- Fusion Branch ---
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
