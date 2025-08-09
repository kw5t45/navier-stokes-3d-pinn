import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def predict_on_xy_plane(model, t, z=0.5, resolution=50):
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = torch.full_like(X_flat, z)
    T_flat = torch.full_like(X_flat, t)

    inputs = torch.stack([X_flat, Y_flat, Z_flat, T_flat], dim=1)
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    u = outputs[:, 0].reshape(resolution, resolution).cpu().numpy()
    v = outputs[:, 1].reshape(resolution, resolution).cpu().numpy()
    w = outputs[:, 2].reshape(resolution, resolution).cpu().numpy()
    p = outputs[:, 3].reshape(resolution, resolution).cpu().numpy()

    return X.cpu().numpy(), Y.cpu().numpy(), u, v, w, p


def animate_flow(model, z=0.5, t_start=0.0, t_end=1.0, frames=50, resolution=50):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Navier-Stokes Flow at z={z}, t=0')

    # Initial data
    X, Y, u, v, w, p = predict_on_xy_plane(model, torch.tensor(t_start), z, resolution)
    speed = np.sqrt(u ** 2 + v ** 2)

    # Pressure colormap background
    pressure_plot = ax.imshow(p, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(pressure_plot, ax=ax)
    cbar.set_label('Pressure p')

    # Velocity quiver
    skip = (slice(None, None, 3), slice(None, None, 3))  # skip arrows for clarity
    quiver = ax.quiver(X[skip], Y[skip], u[skip], v[skip], scale=5, color='white')

    def update(frame):
        t_val = t_start + frame * (t_end - t_start) / (frames - 1)
        ax.set_title(f'Navier-Stokes Flow at z={z}, t={t_val:.2f}')
        X, Y, u, v, w, p = predict_on_xy_plane(model, torch.tensor(t_val), z, resolution)

        pressure_plot.set_data(p)
        speed = np.sqrt(u ** 2 + v ** 2)
        pressure_plot.set_clim(np.min(p), np.max(p))

        quiver.set_UVC(u[skip], v[skip])
        return pressure_plot, quiver

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=100)
    plt.show()
    return anim

def plot_velocity_field(model, t=0.5, z=0.5, resolution=50):
    # Create grid points on xy plane
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Flatten and prepare input tensor [N,4] for (x,y,z,t)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = torch.full_like(X_flat, z)
    T_flat = torch.full_like(X_flat, t)

    inputs = torch.stack([X_flat, Y_flat, Z_flat, T_flat], dim=1)
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Predict velocity and pressure
    with torch.no_grad():
        outputs = model(inputs)
    u = outputs[:, 0].cpu().numpy().reshape(resolution, resolution)
    v = outputs[:, 1].cpu().numpy().reshape(resolution, resolution)

    # Plot velocity vectors
    plt.figure(figsize=(7,6))
    plt.quiver(X.numpy(), Y.numpy(), u, v, scale=5, pivot='mid', color='blue')
    plt.title(f"Velocity field on XY plane at z={z}, t={t}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
