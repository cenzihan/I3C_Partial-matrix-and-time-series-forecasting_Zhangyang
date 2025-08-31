import torch

def complex_mse(y_true, y_pred, mask, weight_factor):
    """
    Computes the complex Mean Squared Error, with weighted loss for missing parts.

    Args:
        y_true (torch.Tensor): The ground truth tensor.
        y_pred (torch.Tensor): The predicted tensor.
        mask (torch.Tensor): A tensor of the same shape as y_true, with 1s where
                             data was missing and 0s where it was present.
        weight_factor (float): The factor to multiply the loss on the missing parts.
    """
    error = y_true - y_pred
    
    # Explicitly calculate the squared magnitude to ensure the result is a real tensor.
    # |z|^2 = z.real^2 + z.imag^2
    squared_magnitude = error.real**2 + error.imag**2
    
    # Create a weight tensor: `weight_factor` for missing parts, 1.0 for present parts.
    weights = 1.0 + (weight_factor - 1.0) * mask
    
    # Apply the weights
    weighted_squared_error = weights * squared_magnitude
    
    # Return the mean of the weighted error
    return torch.mean(weighted_squared_error)
