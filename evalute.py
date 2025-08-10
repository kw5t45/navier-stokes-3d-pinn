from model import NeuralNetwork
import torch
import numpy as np
# model = NeuralNetwork()  # initialize model exactly like before
# model.load_state_dict(torch.load("navier_stokes_model.pth"))
# model.eval()  # set to evaluation mode
#
# n_per_axis = 10
# coords = torch.linspace(0, 1, n_per_axis)
#
# # generate 3D grid
# x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
# xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # shape [1000, 3]
#
# # append fourth column as zeros
# #points = torch.cat([xyz, torch.zeros(len(xyz), 1)], dim=1)
# points = torch.cat([xyz, torch.full((len(xyz), 1), 0.7)], dim=1)
#
# device = next(model.parameters()).device
# points = points.to(device)
#
# with torch.no_grad():
#     output = model(points)  # output shape [N,4] -> [u,v,w,p]
#
# print("Predicted velocity (u,v,w) and pressure p:\n", output.cpu().numpy())
#
# import matplotlib.pyplot as plt
#
# # Extract u, v, w from model output
# u = output[:, 0].cpu().numpy()
# v = output[:, 1].cpu().numpy()
# w = output[:, 2].cpu().numpy()
#
# # Extract coordinates from points
# x = points[:, 0].cpu().numpy()
# y = points[:, 1].cpu().numpy()
# z = points[:, 2].cpu().numpy()

# # 3D quiver plot
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(x, y, z, u, v, w, length=0.1, normalize=False)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Predicted Velocity Field')
#
# plt.show()
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_vector_field(model_path, n_per_axis=10, n_frames=20, t_start=0.0, t_end=1.0):
    # Load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate spatial grid
    coords = torch.linspace(0, 1, n_per_axis)
    xg, yg, zg = torch.meshgrid(coords, coords, coords, indexing='ij')
    xyz = torch.stack([xg, yg, zg], dim=-1).reshape(-1, 3)  # [N, 3]

    device = next(model.parameters()).device
    xyz = xyz.to(device)

    # Prepare time values
    times = torch.linspace(t_start, t_end, n_frames)

    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    quiver = None

    def update(frame_idx):
        nonlocal quiver
        ax.clear()

        # Prepare points with current time
        t_val = times[frame_idx].item()
        points = torch.cat([xyz, torch.full((len(xyz), 1), t_val, device=device)], dim=1)

        # Predict
        with torch.no_grad():
            output = model(points)

        u = output[:, 0].cpu().numpy()
        v = output[:, 1].cpu().numpy()
        w = output[:, 2].cpu().numpy()

        # True normalization to unit vectors
        mag = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag[mag == 0] = 1e-8  # avoid division by zero
        u /= mag
        v /= mag
        w /= mag

        x = xyz[:, 0].cpu().numpy()
        y = xyz[:, 1].cpu().numpy()
        z = xyz[:, 2].cpu().numpy()

        # Plot quiver with fixed length arrows
        quiver = ax.quiver(x, y, z, u, v, w, length=0.1, normalize=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Predicted Velocity Field at t={t_val:.2f}s')

        return quiver,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    plt.show()

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_xy_slice(model_path, n_per_axis=20, n_frames=20, t_start=0.0, t_end=1.0, fixed_z=0.5):
    # Load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = next(model.parameters()).device

    # Generate 2D grid in x,y
    coords = torch.linspace(0, 1, n_per_axis)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    xy = torch.stack([xg, yg], dim=-1).reshape(-1, 2)  # [N,2]

    # Add fixed z and varying time to create input points [N,4]
    fixed_z_tensor = torch.full((len(xy), 1), fixed_z)

    times = torch.linspace(t_start, t_end, n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    quiver = None

    def update(frame_idx):
        nonlocal quiver
        ax.clear()

        t_val = times[frame_idx].item()
        t_tensor = torch.full((len(xy), 1), t_val)

        points = torch.cat([xy, fixed_z_tensor, t_tensor], dim=1).to(device)  # [N,4]

        with torch.no_grad():
            output = model(points)  # [N,4]

        u = output[:, 0].cpu().numpy()
        v = output[:, 1].cpu().numpy()

        X = xy[:, 0].cpu().numpy()
        Y = xy[:, 1].cpu().numpy()

        quiver = ax.quiver(X, Y, u, v, angles='xy', scale_units='xy', color='blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Velocity field slice at z={fixed_z}, t={t_val:.2f}')

        return quiver,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
    plt.show()


def animate_velocity_magnitude(model_path, n_per_axis=100, n_frames=60,
                                t_start=0.0, t_end=1.0, fixed_z=0.5):
    # Load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = next(model.parameters()).device

    # Generate 2D grid in x,y
    coords = torch.linspace(0, 1, n_per_axis)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    xy = torch.stack([xg, yg], dim=-1).reshape(-1, 2)  # [N, 2]
    fixed_z_tensor = torch.full((len(xy), 1), fixed_z)

    # Time values
    times = torch.linspace(t_start, t_end, n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Initial image
    im = ax.imshow(np.zeros((n_per_axis, n_per_axis)), origin='lower',
                   extent=(0, 1, 0, 1), cmap='plasma', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocity Magnitude')

    def update(frame_idx):
        t_val = times[frame_idx].item()
        t_tensor = torch.full((len(xy), 1), t_val)

        points = torch.cat([xy, fixed_z_tensor, t_tensor], dim=1).to(device)  # [N,4]

        with torch.no_grad():
            output = model(points)  # [N,4]

        u = output[:, 0].cpu().numpy()
        v = output[:, 1].cpu().numpy()
        w = output[:, 2].cpu().numpy()
        # Velocity magnitude
        mag = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        mag_grid = mag.reshape(n_per_axis, n_per_axis)

        im.set_data(mag_grid)
        im.set_clim(vmin=mag_grid.min(), vmax=mag_grid.max())

        ax.set_title(f'Velocity magnitude at z={fixed_z}, t={t_val:.2f}')
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
    plt.show()

animate_velocity_magnitude("navier_stokes_model.pth", n_per_axis=200, n_frames=60, fixed_z=0.00)
# animate_xy_slice("navier_stokes_model.pth", n_per_axis=20, n_frames=60, fixed_z=0.05)
# animate_vector_field("navier_stokes_model.pth", n_per_axis=10, n_frames=15)
