import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here

    x=torch.tensor(features, dtype=torch.float32)
    y=torch.tensor(labels,dtype=torch.float32).view(-1,1)
    weight=torch.tensor(initial_weights,dtype=torch.float32, requires_grad=True)
    bias=torch.tensor(initial_bias,dtype=torch.float32,requires_grad=True)

    mse_values=[]

    for _ in range(epochs):
        predictions=torch.sigmoid(x @ weight+bias).view(-1,1)
        loss=F.mse_loss(predictions,y)
        mse_values.append(round(loss.item(),4))

        loss.backward()

        with torch.no_grad():
            weight-=learning_rate*weight.grad
            bias-=learning_rate*bias.grad
        weight.grad.zero_()
        bias.grad.zero_()

    
    update_weight=[round(w,4) for w in weight.detach().tolist()]
    update_bias=round(bias.detach().item(),4)

    return update_weight, update_bias, mse_values


    pass