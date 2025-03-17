from control.utils import rk4, jacobian

import torch
from torch import nn, Tensor

class Dynamics(nn.Module):
    def __init__(self, dt: float):
        super().__init__()
        self.dt = dt

    def f(self, x: Tensor, u: Tensor):
        raise NotImplementedError

    def step(self, x: Tensor, *u: Tensor, return_grad: bool = False):
        u_was_none = False

        if not u:
            u_was_none = True
            u = torch.zeros(self.control_dim, device = x.device)#.squeeze()
        else:
            u = u[0]  # Unpack tuple from: *u = (u,)

        if return_grad:
            x = x.clone().requires_grad_()
            u = u.clone().requires_grad_()

        x_ = x + rk4(self.f, x, u, dt = self.dt)

        if return_grad:
            Jx = jacobian(x_, x)
            Ju = jacobian(x_, u)

            x_ = x_.detach()
            Jx = Jx.detach()
            Ju = Ju.detach()

            if u_was_none:
                return x_, Jx
            else:
                return x_, Jx, Ju
        else:
            return x_
        
    def unroll(self, x0: Tensor, T: int, u: Tensor = None):
        device = x0.device
        
        ## if no control sequence is provided then default to zero control.
        if u == None:
            u = torch.zeros(T, self.control_dim, device = device)
            
        xt = x0.clone()
        ## dimentions of state and control
        dx = x0.numel()

        X = torch.zeros(dx, T + 1, device = device)

        for t in range(T):
            X[:, t] = xt
            xt = self.step(xt, u[t, :])

        X[:, -1] = xt
        return X