import torch

def complex_mse(y_pred, y_true):
    """
    Computes the Mean Squared Error for complex-valued tensors.
    
    The loss is calculated as the mean of the squared differences
    of both the real and imaginary parts.
    
    Args:
        y_pred (torch.Tensor): The predicted tensor. A complex tensor.
        y_true (torch.Tensor): The ground truth tensor. A complex tensor.
        
    Returns:
        torch.Tensor: A scalar tensor representing the loss.
    """
    # For complex numbers, the squared absolute difference |y_true - y_pred|^2
    # is equivalent to (y_true_real - y_pred_real)^2 + (y_true_imag - y_pred_imag)^2.
    error = y_true - y_pred
    
    # Calculate the mean of the squared magnitude of the error.
    loss = torch.mean(torch.abs(error)**2)
    
    return loss
