from control.dynamics import Dynamics

import torch
from torch import Tensor
import matplotlib.pyplot as plt

class Oscillation(Dynamics):
    def __init__(self, omega: float, drift: float, dt: float = 0.1):
        super().__init__(dt)

        self.A = torch.tensor([
            [0.0, 1.0],
            [-omega**2, -2*drift*omega]
        ])

        self.B = torch.tensor([[0.0], [1.0]])

    def f(self, x: Tensor, u: Tensor):
        return self.A @ x + self.B @ u
    
    @property
    def control_dim(self):
        return 1
    
def riccati(A: Tensor, B:Tensor, Q: Tensor, R: Tensor, P: Tensor):
    return Q + A.T @ P @ A - A.T @ P @ B @ torch.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

if __name__ == '__main__':
    dyn = Oscillation(omega = 0.0, drift = -1.0)
    x0 = torch.zeros(2)

    dx = x0.numel()
    I = torch.eye(dx)
    A = I + dyn.A * dyn.dt
    B = dyn.B * dyn.dt

    Q = torch.eye(dx)
    R = torch.eye(dyn.control_dim)

    T = 100

    P = Q.clone()
    K = []
    for t in reversed(range(T)):
        k = torch.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        K.insert(0, k)
        P = riccati(A, B, Q, R, P)

    # K = torch.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # T = 100

    xt = x0.clone()
    X = [xt]

    x_goal = torch.tensor([0.5, 0.0])

    for i in range(T):
        u = -K[t] @ (xt - x_goal)
        xt = dyn.step(xt, u)
        X.append(xt)

    X = torch.stack(X).T
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(X[0, :], label = 'Position')
    ax.plot(X[1, :], label = 'Velocity')
    ax.legend()
    plt.show()