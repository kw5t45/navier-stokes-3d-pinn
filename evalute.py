from model import NeuralNetwork
import torch
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

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


def animate_particles(model_path, n_per_axis=100, n_frames=200, dt=0.01,
                      fixed_z=0.5, t_start=0.0, save=False, save_path="particles.gif"):
    # Load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = next(model.parameters()).device

    # Initial particle positions (grid)
    coords = torch.linspace(0, 1, n_per_axis)
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    xy = torch.stack([xg, yg], dim=-1).reshape(-1, 2)  # [N, 2]
    fixed_z_tensor = torch.full((len(xy), 1), fixed_z)

    # Convert to tensor on device
    positions = torch.cat([xy, fixed_z_tensor], dim=1).to(device)  # [N,3]

    # Random colors for particles (fixed)
    colors = cm.hsv(xy[:, 0])  # hue depends on x position

    # Time tracking
    t_val = t_start

    # Figure/axes setup
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(positions[:, 0].cpu().numpy(),
                    positions[:, 1].cpu().numpy(),
                    c=colors, s=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f"t = {t_val:.2f}")

    def update(frame_idx):
        nonlocal t_val, positions
        t_tensor = torch.full((len(positions), 1), t_val, device=device)

        # Build full (x, y, z, t) input
        pts_4d = torch.cat([positions, t_tensor], dim=1)  # [N,4]

        with torch.no_grad():
            output = model(pts_4d)  # [N,4]
        u = output[:, 0]
        v = output[:, 1]
        # w = output[:, 2]  # Ignored for 2D slice

        # Update positions
        positions[:, 0] += u * dt
        positions[:, 1] += v * dt

        # Optional: keep inside domain
        positions[:, 0] = torch.clamp(positions[:, 0], 0, 1)
        positions[:, 1] = torch.clamp(positions[:, 1], 0, 1)

        # Update scatter plot
        sc.set_offsets(positions[:, :2].cpu().numpy())
        t_val += dt
        ax.set_title(f"t = {t_val:.2f}")
        return [sc]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=60, blit=False)

    if save:
        anim.save(save_path, writer=PillowWriter(fps=60))
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

#animate_particles('s=0.2_nu=1e-06_lr1=0.0005_lr2=0.00075.pth', 200, 200, 0.01, 0.25, save=True)
# animate_velocity_magnitude("s=0.2_nu=1e-06_lr1=0.0005_lr2=0.00075.pth", n_per_axis=200, n_frames=60, fixed_z=0.5)
animate_xy_slice("s=0.2_nu=1e-06_lr1=0.0005_lr2=0.00075.pth", n_per_axis=80, n_frames=60, fixed_z=0.125)
#animate_vector_field("s=0.2_nu=1e-06_lr1=0.0005_lr2=0.00075.pth", n_per_axis=10, n_frames=15)
