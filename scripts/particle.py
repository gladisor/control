from control.dynamics import Dynamics

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Particle(Dynamics):
    def __init__(self, alpha: float = 1.0, dt: float = 0.1):
        super().__init__(dt)
        self.alpha = alpha

    @property
    def control_dim(self):
        return 2
    
    def acceleration(self, u: Tensor):
        ## original (uncontrollable) dynamics
        # p1_dot = self.alpha * torch.cos(u[0])
        # p2_dot = self.alpha * torch.sin(u[0])

        ## modified (controllable) dynamics
        p1_dot = u[0]
        p2_dot = u[1]
        return p1_dot, p2_dot

    def f(self, x, u):
        p1, p2, q1, q2 = x

        p1_dot, p2_dot = self.acceleration(u)
        q1_dot = p1
        q2_dot = p2

        return torch.stack([
            p1_dot,
            p2_dot,
            q1_dot,
            q2_dot
        ])
    
    def animate(self, X: Tensor, u = None, x_goal: Tensor = None):
        p1_values, p2_values = X[0, :], X[1, :]
        q1_values, q2_values = X[2, :], X[3, :]

        min_xy = torch.min(q1_values.min(), q2_values.min())
        max_xy = torch.max(q1_values.max(), q2_values.max())
        
        fig, ax = plt.subplots()
        ax.set_aspect(1.0)
        fig.tight_layout()

        ax.set_xlim(min_xy - 1, max_xy + 1)
        ax.set_ylim(min_xy - 1, max_xy + 1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Particle Motion')

        ax.scatter(x_goal[2], x_goal[3], color = 'green', alpha = 0.5, label = 'Goal')
        ax.legend()
        
        particle, = ax.plot([], [], 'bo', markersize = 8)  # Blue particle
        tracer, = ax.plot([], [], 'r-', alpha=0.5, lw = 1)  # Red fading tracer
        arrow = ax.arrow(0.0, 0.0, 0.0, 0.0, width = 0.01)
        
        trail_length = 20  # Number of previous points to keep in the tracer
        
        def init():
            particle.set_data([], [])
            tracer.set_data([], [])
            arrow.set_data(x = 0.0, y = 0.0, dx = 0.0, dy = 0.0)
            return particle, tracer, arrow
        
        def update(frame):
            particle.set_data([q1_values[frame]], [q2_values[frame]])  # Ensure these are lists
            start = max(0, frame - trail_length)
            tracer.set_data(q1_values[start:frame + 1], q2_values[start:frame + 1])  # Include frame + 1 for proper slicing

            if u == None:
                arrow.set_data(x = q1_values[frame], y = q2_values[frame], dx = p1_values[frame], dy = p2_values[frame])
            else:
                p1_dot, p2_dot = self.acceleration(u[frame])
                p1_dot, p2_dot = p1_dot / dyn.alpha, p2_dot / dyn.alpha
                arrow.set_data(x = q1_values[frame], y = q2_values[frame], dx = p1_dot, dy = p2_dot)
            return particle, tracer, arrow
        
        ani = animation.FuncAnimation(fig, update, frames = X.shape[1] - 1, init_func = init, blit = True, interval = 20)
        writer = animation.FFMpegWriter(fps = 60)
        ani.save('particle.mp4', writer = writer)
        return

if __name__ == '__main__':
    dyn = Particle(alpha = 0.1)
    dx = 4
    du = dyn.control_dim
    Q = torch.eye(dx)
    R = torch.eye(du)
    x0 = torch.zeros(dx)

    T = 100
    x_goal = torch.tensor([0.0, 0.0, -1.0, -1.0])
    xt = x0.clone()

    u = torch.zeros(T, dyn.control_dim)

    A = []
    B = []
    for t in range(T):
        xt, Jx, Ju = dyn.step(xt, u[t, :], return_grad = True)
        A.append(Jx)
        B.append(Ju)

    P = Q.clone()
    K = []
    for t in reversed(range(T)):
        k = -torch.linalg.inv(R + B[t].T @ P @ B[t]) @ B[t].T @ P @ A[t]
        K.insert(0, k)
        P = Q + A[t].T @ P @ A[t] - A[t].T @ P @ B[t] @ torch.linalg.inv(R + B[t].T @ P @ B[t]) @ B[t].T @ P @ A[t]

    X = [x0.clone()]
    xt = x0.clone()
    for t in range(T):
        u = K[t] @ (xt - x_goal)
        xt = dyn.step(xt, u)
        X.append(xt)

    X = torch.stack(X).T
    dyn.animate(X, x_goal = x_goal)

