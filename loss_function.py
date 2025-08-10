import torch
import math


# first navier stokes equation
def incompressibility_loss(model, collocation_points):
    collocation_points.requires_grad_(True)
    outputs = model(collocation_points)  # [N,4]: u,v,w,p

    u = outputs[:, 0]
    v = outputs[:, 1]
    w = outputs[:, 2]  # we dont need pressure output so we stop here

    grads = {}
    grads['u_x'] = torch.autograd.grad(u, collocation_points,
                                       grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0][:, 0]
    grads['v_y'] = torch.autograd.grad(v, collocation_points,
                                       grad_outputs=torch.ones_like(v),
                                       create_graph=True)[0][:, 1]
    grads['w_z'] = torch.autograd.grad(w, collocation_points,
                                       grad_outputs=torch.ones_like(w),
                                       create_graph=True)[0][:, 2]

    divergence = grads['u_x'] + grads['v_y'] + grads['w_z']
    loss = torch.mean(divergence ** 2)
    return loss


# pressure anchor point loss

def anchor_loss(model):
    anchor_point = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    output = model(anchor_point)  # [1,4]: u,v,w,p
    p_anchor = output[0, 3]  # pressure scalar
    loss = (p_anchor - 0.0) ** 2
    return loss


# no slip boundary condition loss

def boundary_loss(model, collocation_points):
    """
    collocation_points: [N,4] tensor with columns (x,y,z,t)
    Boundary points are those on the walls of the cube in spatial domain:
    x=0 or 1, y=0 or 1, or z=0 or 1.
    """

    # Extract spatial coords (ignore t for boundary condition)
    x = collocation_points[:, 0]
    y = collocation_points[:, 1]
    z = collocation_points[:, 2]

    # Create a boolean mask for boundary points
    tol = 0.01  # tolerance for floating point comparison
    on_boundary = ( (torch.abs(x) < tol) | (torch.abs(x - 1) < tol) |
                    (torch.abs(y) < tol) | (torch.abs(y - 1) < tol) |
                    (torch.abs(z) < tol) | (torch.abs(z - 1) < tol) )

    boundary_points = collocation_points[on_boundary]

    if boundary_points.shape[0] == 0:
        # No boundary points found, return zero loss or handle as needed
        return torch.tensor(0.0, device=collocation_points.device)

    outputs = model(boundary_points)  # [N_b, 4]: u,v,w,p
    u_pred = outputs[:, :3]            # predicted velocity (u,v,w)
    u_target = torch.zeros_like(u_pred)  # no-slip condition: velocity = 0
    loss = torch.mean((u_pred - u_target) ** 2)
    return loss



# for initial loss we first define intial velocity vector field:

def initial_loss(model, collocation_points, sigma):
    t = collocation_points[:, 3]
    tol = 0.1
    mask = (torch.abs(t) < tol)
    initial_points = collocation_points[mask]  # shape [N_initial,4]
    if initial_points.shape[0] == 0:
        return torch.tensor(0.0, device=collocation_points.device)

    outputs = model(initial_points)
    u_pred = outputs[:, :3]  # predicted velocity vector

    # Extract spatial coordinates from initial points (not full collocation points!)
    x = initial_points[:, 0]
    y = initial_points[:, 1]
    z = initial_points[:, 2]

    # Calculate psi only at initial points
    psi = 10*torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / (sigma ** 2))

    # Define true initial velocity at initial points
    u_true = torch.stack([
        - (2 * (y - 0.5) / (sigma ** 2)) * psi,
        (2 * (x - 0.5) / (sigma ** 2)) * psi,
        (2*(z-0.5)/(sigma**2))*psi
    ], dim=1)
    #
    # u_true = torch.stack([
    #     x, y, z
    # ], dim=1)


    loss = torch.mean((u_pred - u_true) ** 2)

    return loss



# finally the pde equation loss

def navier_stokes_pde_loss(model, collocation_points, nu):
    collocation_points.requires_grad_(True)
    outputs = model(collocation_points)
    u, v, w, p = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

    def gradients(output, inputs, idx):
        """
        output: [N] tensor
        inputs: [N, 4] tensor with columns x,y,z,t
        idx: int, which coordinate to differentiate w.r.t (0=x,1=y,2=z,3=t)
        """
        grad = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape: [N,4]

        if grad is None:
            # In case gradient is not connected (allow_unused=True), return zeros
            return torch.zeros_like(output)

        return grad[:, idx]

    # Compute gradients w.r.t each coordinate by passing whole collocation_points and index
    u_x = gradients(u, collocation_points, 0)
    u_y = gradients(u, collocation_points, 1)
    u_z = gradients(u, collocation_points, 2)
    u_t = gradients(u, collocation_points, 3)

    v_x = gradients(v, collocation_points, 0)
    v_y = gradients(v, collocation_points, 1)
    v_z = gradients(v, collocation_points, 2)
    v_t = gradients(v, collocation_points, 3)

    w_x = gradients(w, collocation_points, 0)
    w_y = gradients(w, collocation_points, 1)
    w_z = gradients(w, collocation_points, 2)
    w_t = gradients(w, collocation_points, 3)

    p_x = gradients(p, collocation_points, 0)
    p_y = gradients(p, collocation_points, 1)
    p_z = gradients(p, collocation_points, 2)

    def second_derivative(output, inputs, idx):
        first_deriv = gradients(output, inputs, idx)
        second_deriv = gradients(first_deriv, inputs, idx)
        return second_deriv

    u_xx = second_derivative(u, collocation_points, 0)
    u_yy = second_derivative(u, collocation_points, 1)
    u_zz = second_derivative(u, collocation_points, 2)

    v_xx = second_derivative(v, collocation_points, 0)
    v_yy = second_derivative(v, collocation_points, 1)
    v_zz = second_derivative(v, collocation_points, 2)

    w_xx = second_derivative(w, collocation_points, 0)
    w_yy = second_derivative(w, collocation_points, 1)
    w_zz = second_derivative(w, collocation_points, 2)

    R_u = u_t + u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    R_v = v_t + u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    R_w = w_t + u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)

    loss = torch.mean(R_u ** 2 + R_v ** 2 + R_w ** 2)
    return loss



def total_loss(model, collocation_points, nu=0.00001, simga=0.05, l_pde=0.01, l_incompressibility=0.01, l_ic=0.01,
               l_bc=0.01, l_anchor=0.01):
    total = l_pde * navier_stokes_pde_loss(model, collocation_points, nu) + l_incompressibility * incompressibility_loss(
        model, collocation_points) + l_ic * initial_loss(model, collocation_points, simga) + l_bc*boundary_loss(
        model, collocation_points) + l_anchor*anchor_loss(model)
    return total, navier_stokes_pde_loss(model, collocation_points, nu), incompressibility_loss(
        model, collocation_points), initial_loss(model, collocation_points, simga), boundary_loss(
        model, collocation_points), anchor_loss(model)