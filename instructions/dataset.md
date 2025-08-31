# Espargos-CSI Dataset Introduction

This directory contains information about the dataset used in this project.

## Source

The dataset is a processed subset of the **Espargos-CSI** dataset, originally published for Wi-Fi sensing research.

## Content

The preprocessed data is now stored as `datasets/csi_data.h5` in HDF5 format for PyTorch compatibility.

Each record in the dataset consists of several components, but for this project, we primarily focus on the Channel State Information (CSI).

### CSI Tensor

- **Shape**: `(4, 2, 4, 117)`
- **Data Type**: `complex64`

#### Dimensionality Breakdown:
- **4 (first dimension)**: The number of devices (4 devices total).
- **2**: The number of rows in the receiver antenna matrix.
- **4 (third dimension)**: The number of columns in the receiver antenna matrix.
- **117**: The number of subcarriers used in the Wi-Fi channel (typically corresponding to a 40MHz channel width).

#### Data Structure Interpretation:
- The second and third dimensions `(2, 4)` together form a **receiver antenna matrix** of size 2×4, representing the spatial arrangement of receiving antennas.
- Each device captures CSI measurements across this antenna matrix for all 117 subcarriers.
- The CSI values are complex numbers containing both magnitude and phase information for each antenna-subcarrier combination.

## Temporal Characteristics

- **Total Records**: 186,879
- **Ordering**: The records are strictly sorted by timestamp in ascending order.
- **Continuity**: The dataset is composed of continuous segments but contains at least one significant time gap of ~0.48 seconds. It is not a single, perfectly continuous time series.
- **Average Sampling Rate**: Approximately 97.6 Hz (~10.24 ms per sample).
