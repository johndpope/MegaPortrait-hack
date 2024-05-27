import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Create tensors for x, y, and z coordinates
x = torch.linspace(0, 10, 50)
y = torch.linspace(0, 10, 50)
X, Y = torch.meshgrid(x, y)
Z1 = torch.sin(X) + torch.randn(X.shape) * 0.2
Z2 = torch.sin(X + 1.5) + torch.randn(X.shape) * 0.2
Z3 = Z1 + Z2

# Create a figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the dots with quiver for direction/flow
q1 = ax.quiver(X, Y, Z1, Z1, Z1, Z1, length=0.1, normalize=True, cmap='viridis', label='x_e,k')
q2 = ax.quiver(X, Y, Z2, Z2, Z2, Z2, length=0.1, normalize=True, cmap='plasma', label='R_d+c,k')
q3 = ax.quiver(X, Y, Z3, Z3, Z3, Z3, length=0.1, normalize=True, cmap='inferno', label='R_d+c,k + t_d')

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('PyTorch Tensor Plot (3D)')

# Add a legend
ax.legend()

# Display the plot
plt.show()