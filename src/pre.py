import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm

raw_dataset = tf.data.TFRecordDataset(["datasets/espargos-0005-meanders-nw-se.tfrecords",])
feature_description = {
	"csi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
	"mac": tf.io.FixedLenFeature([], tf.string, default_value = ''),
	"pos": tf.io.FixedLenFeature([], tf.string, default_value = ''),
	"rssi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
	"time": tf.io.FixedLenFeature([], tf.string, default_value = ''),
}
			
def record_parse_function(proto):
	record = tf.io.parse_single_example(proto, feature_description)

	# Channel coefficients for all antennas, over all subcarriers, complex-valued
	csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type = tf.complex64), (4, 2, 4, 117))

	# MAC address of transmitter
	mac = record["mac"]

	# Position of transmitter determined by a tachymeter pointed at a prism mounted on top of the antenna, in meters (X / Y / Z coordinates)
	pos = tf.ensure_shape(tf.io.parse_tensor(record["pos"], out_type = tf.float64), (3))

	# Received signal strength indicator (in dB) for all antennas
	rssi = tf.ensure_shape(tf.io.parse_tensor(record["rssi"], out_type = tf.float32), (4, 2, 4))

	# Timestamp of measurement, seconds since UNIX epoch
	time = tf.ensure_shape(tf.io.parse_tensor(record["time"], out_type = tf.float64), ())

	return csi, mac, pos, rssi, time
			
dataset = raw_dataset.map(record_parse_function, num_parallel_calls = tf.data.experimental.AUTOTUNE)

# Optional: Cache dataset in RAM for faster training
dataset = dataset.cache()

# Inspect the structure of one element from the dataset
print("--- Data Structure ---")
for csi, mac, pos, rssi, time in dataset.take(1):
    print(f"CSI      - Shape: {csi.shape}, DType: {csi.dtype}")
    print(f"MAC      - Shape: {mac.shape}, DType: {mac.dtype}")
    print(f"Position - Shape: {pos.shape}, DType: {pos.dtype}")
    print(f"RSSI     - Shape: {rssi.shape}, DType: {rssi.dtype}")
    print(f"Time     - Shape: {time.shape}, DType: {time.dtype}")

# Save the preprocessed dataset (TF format)
save_path = "datasets/predata"
print(f"\nSaving preprocessed data to: {save_path}")
dataset.save(save_path)
print("Data saved successfully in TensorFlow format.")

# --- New: Extract and save CSI data to HDF5 for PyTorch ---
print("\n--- Extracting CSI data for PyTorch compatibility ---")
csi_tensors = []
# Use tqdm for a progress bar as this can be slow
for csi, _, _, _, _ in tqdm(dataset, desc="Extracting CSI tensors"):
    csi_tensors.append(csi.numpy())

# Stack all tensors into a single NumPy array
csi_numpy_array = np.stack(csi_tensors, axis=0)

# Save to HDF5
h5_save_path = "datasets/csi_data.h5"
print(f"Saving CSI data as a NumPy array to: {h5_save_path}")
with h5py.File(h5_save_path, 'w') as f:
    f.create_dataset('csi', data=csi_numpy_array)

print(f"Shape of saved CSI data: {csi_numpy_array.shape}")
print("HDF5 file created successfully for PyTorch.")