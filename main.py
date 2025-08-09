import torch
import torch.nn as nn
import torch.optim as optim
from animations import predict_on_xy_plane, animate_flow, plot_velocity_field
import numpy as np
import matplotlib.pyplot as plt

from pyDOE import lhs
from loss_function import total_loss

# MLP decleration
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 4)
            )

    def forward(self, x):
        return self.model(x)

# collocation points


def generate_lhs_collocation_points(n_points):
    print("Generating collocation points ...")
    # R4 since x y z t
    lhs_samples = lhs(4, samples=n_points, criterion='maximin')
    # lhs() returns points in [0,1]^4, so no scaling needed
    collocation_points = torch.tensor(lhs_samples, dtype=torch.float32)
    return collocation_points

def sample_boundary_points(n_points_per_face=330):
    boundary_points = []
    # for each face: fix one spatial coordinate to 0 or 1
    # and sample the other two spatial coordinates + time uniformly
    for fixed_dim, fixed_val in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]:
        # Sample 3 variables: two spatial dims + time
        pts = lhs(3, samples=n_points_per_face, criterion='maximin')
        # creating a full 4D tensor with the fixed dimension set
        for pt in pts:
            full_pt = [0.0, 0.0, 0.0, 0.0]
            spatial_dims = [0,1,2]
            spatial_dims.remove(fixed_dim)
            full_pt[fixed_dim] = fixed_val
            full_pt[spatial_dims[0]] = pt[0]
            full_pt[spatial_dims[1]] = pt[1]
            full_pt[3] = pt[2]  # time coordinate
            boundary_points.append(full_pt)
    return torch.tensor(boundary_points, dtype=torch.float32)

def sample_initial_points(n_points=2000):
    # Sample uniformly in spatial domain at t=0
    pts = lhs(3, samples=n_points, criterion='maximin')
    initial_points = []
    for pt in pts:
        full_pt = [pt[0], pt[1], pt[2], 0.0]
        initial_points.append(full_pt)
    return torch.tensor(initial_points, dtype=torch.float32)


n_collocation_r4 = 20000
n_collocation_initial = 1000
n_collocation_boundary = 1000//6


boundary_pts = sample_boundary_points(n_collocation_boundary)
initial_pts = sample_initial_points(n_collocation_initial)
collocation_points = generate_lhs_collocation_points(n_collocation_r4)

print(boundary_pts.shape)  # Should be about [6 * input ,4]
print(initial_pts.shape)   # Should be [200,4]
print(collocation_points.shape)  # Should be [20000, 4]

all_points = torch.cat([boundary_pts, initial_pts, collocation_points], dim=0)

print("Boundary points shape:", boundary_pts.shape)
print("Initial points shape:", initial_pts.shape)
print("Collocation points shape:", collocation_points.shape)
print("All points concatenated shape:", all_points.shape)


# model training

model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop example
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    losses = total_loss(model, all_points, nu=0.0001, simga=0.05,
                      l_pde=1, l_incompressibility=1,
                      l_ic=10, l_bc=1, l_anchor=1)
    loss = losses[0]
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # print(f"Epoch {epoch}, Total Loss: {losses[0]}, PDE Loss: {losses[1]}, Incompressibility Loss: {losses[2]}, "
        #       f"Initial Loss: {losses[3]}, Boundary Loss: {losses[4]}, Anchor Loss: {losses[5]} ")
        print(f"Epoch {epoch} Total {losses[0]} PDE {losses[1]} Inc. {losses[2]}"
              f"I {losses[3]} B {losses[4]} A {losses[5]} ")
torch.save(model.state_dict(), "navier_stokes_model.pth")


