import torch
from torch import Tensor

def rk4(f: callable, x: Tensor, *u: Tensor, dt: float = 0.1) -> Tensor:
    k1 = f(x, *u)
    k2 = f(x + 0.5 * k1 * dt, *u)
    k3 = f(x + 0.5 * k2 * dt, *u)
    k4 = f(x + k3 * dt, *u)
    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return dx * dt

def grad(outputs: Tensor, inputs: Tensor) -> Tensor:
    '''
    Computes the partial derivative of 
    an output with respect to an input.
    '''
    return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs = torch.ones_like(outputs, device = outputs.device), 
        create_graph=True,
    )[0]

def jacobian(output: Tensor, input: Tensor) -> Tensor:
    '''
    Computes the jacobian of a function.
    '''
    ## should only work for vector functions
    assert output.ndim == 1

    J = torch.stack([
        grad(output[i], input) for i in range(output.shape[0])
    ])

    return J