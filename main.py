import torch
import torch.optim as optim
from tqdm import tqdm
from model import NeuralNetwork


from pyDOE import lhs
from loss_function import *


# collocation points


def generate_lhs_collocation_points(n_points):
    print("Generating collocation points ...")
    # R4 since x y z t
    lhs_samples = lhs(4, samples=n_points)
    # lhs() returns points in [0,1]^4, so no scaling needed
    collocation_points = torch.tensor(lhs_samples, dtype=torch.float32)
    return collocation_points


def sample_boundary_points(n_points_per_face=330):
    print("Sampling boundary points")
    boundary_points = []
    # for each face: fix one spatial coordinate to 0 or 1
    # and sample the other two spatial coordinates + time uniformly
    for fixed_dim, fixed_val in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]:
        # Sample 3 variables: two spatial dims + time
        pts = lhs(3, samples=n_points_per_face)
        # creating a full 4D tensor with the fixed dimension set
        for pt in pts:
            full_pt = [0.0, 0.0, 0.0, 0.0]
            spatial_dims = [0, 1, 2]
            spatial_dims.remove(fixed_dim)
            full_pt[fixed_dim] = fixed_val
            full_pt[spatial_dims[0]] = pt[0]
            full_pt[spatial_dims[1]] = pt[1]
            full_pt[3] = pt[2]  # time coordinate
            boundary_points.append(full_pt)
    return torch.tensor(boundary_points, dtype=torch.float32)


def sample_initial_points(n_points=2000):
    # Sample uniformly in spatial domain at t=0
    print("Sampling Initial points...")
    pts = lhs(3, samples=n_points)
    initial_points = []
    for pt in pts:
        full_pt = [pt[0], pt[1], pt[2], 0.0]
        initial_points.append(full_pt)
    return torch.tensor(initial_points, dtype=torch.float32)


n_collocation_r4 = 30_000
n_collocation_initial = 10000 # 10000
n_collocation_boundary = 1000 # 600000 // 6 # total points -  points per surface

boundary_pts = sample_boundary_points(n_collocation_boundary)
initial_pts = sample_initial_points(n_collocation_initial)
collocation_points = generate_lhs_collocation_points(n_collocation_r4)

# print(boundary_pts.shape)  # Should be about [6 * input ,4]
# print(initial_pts.shape)  # Should be [200,4]
# print(collocation_points.shape)  # Should be [20000, 4]

all_points = torch.cat([boundary_pts, initial_pts, collocation_points], dim=0)
# shuffle
perm = torch.randperm(all_points.size(0))

# Apply permutation to shuffle rows
all_points = all_points[perm]
# print("Boundary points shape:", boundary_pts.shape)
# print("Initial points shape:", initial_pts.shape)
# print("Collocation points shape:", collocation_points.shape)
print("All points concatenated shape:", all_points.shape)

# model training

# ic
model = NeuralNetwork()
num_epochs = 2000
lr_1 = 0.0005
sigma = 0.2
nu = 0.000001
lr_2=0.00075


optimizer = optim.Adam(model.parameters(), lr=lr_1)

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    loss = initial_loss(model, initial_pts, sigma)
    loss.backward()
    optimizer.step()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: No grad for {name}")

    if epoch % 100 == 0:
        print(f'IC loss: {loss.item():.4f}')


optimizer = optim.Adam(model.parameters(), lr=lr_2)

num_epochs = 100 #1500
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    pde = navier_stokes_pde_loss(model, all_points, nu)
    inc = incompressibility_loss(model, all_points)
    anc = anchor_loss(model)
    bc = boundary_loss(model, all_points)
    ic_loss = initial_loss(model, all_points, sigma)

    # Ramp factor grows from 0 → 1 over first 1000 steps
    alpha = min(1.0, epoch / 1000)

    loss = 0.1 * ic_loss + alpha * (pde + inc + 1e-4 * anc + bc)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"TOTAL (weighted): {loss.item():.4f} | IC: {ic_loss.item():.4f} | PDE: {pde.item():.4f} | INCOMP.: {inc.item():.4f} | BC: {bc.item():.4f}")

torch.save(model.state_dict(), f's={sigma}_nu={nu}_lr1={lr_1}_lr2={lr_2}.pth')