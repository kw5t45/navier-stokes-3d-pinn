from main import NeuralNetwork
import torch

model = NeuralNetwork()  # initialize model exactly like before
model.load_state_dict(torch.load("navier_stokes_model.pth"))
model.eval()  # set to evaluation mode

# Example input point(s)
# Single point:
point = torch.tensor([[0.3, 0.7, 0.5, 0.1]], dtype=torch.float32)  # shape [1,4]

# Or multiple points:
points = torch.tensor([
    [0.3, 0.7, 0.5, 0.1],
    [0.6, 0.2, 0.5, 0.3],
    [0.1, 0.1, 0.5, 0.0]
], dtype=torch.float32)  # shape [N,4]

device = next(model.parameters()).device
points = points.to(device)

with torch.no_grad():
    output = model(points)  # output shape [N,4] -> [u,v,w,p]

print("Predicted velocity (u,v,w) and pressure p:\n", output.cpu().numpy())
