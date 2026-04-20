import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
# ======================================================
# 1. Device & parameters
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

eps = 0.01
pi = torch.tensor(np.pi, device=device)

print(f"Using device: {device}")
print(f"epsilon = {eps:.4f}")

# ======================================================
# 2. Pseudo-sequence generator (spatial in x)
# ======================================================
def pseudo_sequence(xt, dx):
    """
    Input:  xt = [N,3] -> (x,y,t)
    Output: [N,3,3]    -> [(x-dx,y,t), (x,y,t), (x+dx,y,t)]
    """
    x = xt[:, 0:1]
    y = xt[:, 1:2]
    t = xt[:, 2:3]
    return torch.stack([
        torch.cat([x - dx, y, t], dim=1),
        torch.cat([x,       y, t], dim=1),
        torch.cat([x + dx, y, t], dim=1)
    ], dim=1)  # [N,3,3]

# ======================================================
# 3. Transformer block (encoder only)
# ======================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Tanh(),
            nn.Linear(256, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        return self.norm2(x + h)

# ======================================================
# 4. Trans-PINN model (same structure, 3D input)
# ======================================================
class TransPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 64)  # (x,y,t) -> 64
        self.transformer = TransformerBlock(64, 4)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, xt, dx):
        # xt : [N,3] = (x,y,t)
        seq = pseudo_sequence(xt, dx)   # [N,3,3]
        h = self.embed(seq)             # [N,3,64]
        h = self.transformer(h)         # [N,3,64]
        h = h[:, 1, :]                  # center token [N,64]
        return self.mlp(h)              # [N,1]

model = TransPINN().to(device)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

# ======================================================
# 5. Data: 2D space + time grid
# ======================================================
Nx, Ny, Nt = 32, 32, 30   # you can refine these

x = torch.linspace(0.0, 1.0, Nx)
y = torch.linspace(0.0, 1.0, Ny)
t = torch.linspace(0.0, 1.0, Nt)

X, Y, T = torch.meshgrid(x, y, t, indexing="ij")  # [Nx,Ny,Nt]
res = torch.cat([
    X.reshape(-1,1),
    Y.reshape(-1,1),
    T.reshape(-1,1)
], dim=1)  # [N_res, 3]
res = res.to(device).requires_grad_(True)

dx = x[1] - x[0]
dy = y[1] - y[0]

# Initial condition: t = 0, u(x,y,0) = 0.05 sin(pi x) sin(pi y)
X_ic, Y_ic = torch.meshgrid(x, y, indexing="ij")
x_ic = X_ic.reshape(-1,1).to(device)
y_ic = Y_ic.reshape(-1,1).to(device)
t_ic = torch.zeros_like(x_ic)
ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
u_ic = 0.05 * torch.sin(pi * x_ic) * torch.sin(pi * y_ic)

# ======================================================
# 5b. Boundary conditions: u=0 on all boundaries
# ======================================================
Nt_bc = 50
Ny_bc = 50
Nx_bc = 50

t_bc = torch.linspace(0.0, 1.0, Nt_bc).to(device)

# x = 0 and x = 1
y_vals = torch.linspace(0.0, 1.0, Ny_bc).to(device)
Yb, Tb = torch.meshgrid(y_vals, t_bc, indexing="ij")   # [Ny_bc,Nt_bc]

X0 = torch.zeros_like(Yb)                              # x=0
bc_x0 = torch.cat([
    X0.reshape(-1,1),
    Yb.reshape(-1,1),
    Tb.reshape(-1,1)
], dim=1)  # [Ny_bc*Nt_bc,3]

X1 = torch.ones_like(Yb)                               # x=1
bc_x1 = torch.cat([
    X1.reshape(-1,1),
    Yb.reshape(-1,1),
    Tb.reshape(-1,1)
], dim=1)

# y = 0 and y = 1
x_vals = torch.linspace(0.0, 1.0, Nx_bc).to(device)
Xb, Tb2 = torch.meshgrid(x_vals, t_bc, indexing="ij")  # [Nx_bc,Nt_bc]

Y0 = torch.zeros_like(Xb)                              # y=0
bc_y0 = torch.cat([
    Xb.reshape(-1,1),
    Y0.reshape(-1,1),
    Tb2.reshape(-1,1)
], dim=1)

Y1 = torch.ones_like(Xb)                               # y=1
bc_y1 = torch.cat([
    Xb.reshape(-1,1),
    Y1.reshape(-1,1),
    Tb2.reshape(-1,1)
], dim=1)

bc = torch.cat([bc_x0, bc_x1, bc_y0, bc_y1], dim=0).to(device)

# ======================================================
# 6. Loss function for 2D Allen–Cahn
# ======================================================
def pinn_loss():
    # Residual points
    u = model(res, dx)  # [N_res,1]

    grads = torch.autograd.grad(u.sum(), res, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    # second derivatives
    grads2 = torch.autograd.grad(u_x.sum(), res, create_graph=True)[0]
    u_xx = grads2[:, 0:1]
    grads3 = torch.autograd.grad(u_y.sum(), res, create_graph=True)[0]
    u_yy = grads3[:, 1:2]

    # PDE residual: u_t = eps*(u_xx+u_yy) - u^3 + u
    f = u_t - eps*(u_xx + u_yy) + u**3 - u
    loss_res = torch.mean(f**2)

    # IC loss
    u_ic_pred = model(ic, dx)
    loss_ic = torch.mean((u_ic_pred - u_ic)**2)

    # BC loss: u=0 on all boundaries
    u_bc_pred = model(bc, dx)
    loss_bc = torch.mean(u_bc_pred**2)

    return loss_res + loss_ic + loss_bc

# ======================================================
# 7. Training
# ======================================================
print("Training Trans-PINN on 2D Allen–Cahn (Adam)...")
opt = Adam(model.parameters(), lr=1e-3)

for _ in tqdm(range(8000)):
    opt.zero_grad()
    loss = pinn_loss()
    loss.backward()
    opt.step()

print("Polishing with L-BFGS...")
opt_lbfgs = LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=1200,
    line_search_fn="strong_wolfe"
)

def closure():
    opt_lbfgs.zero_grad()
    l = pinn_loss()
    l.backward()
    return l

opt_lbfgs.step(closure)

# ======================================================
# 8. FDM ground truth for 2D Allen–Cahn
# ======================================================
def allen_cahn_rhs(u_flat, t_val, eps, Nx, Ny, dx, dy):
    """
    u_flat: [Nx*Ny]
    du/dt = eps*(u_xx + u_yy) - u^3 + u
    Zero Dirichlet BC enforced at boundaries.
    """
    u = u_flat.reshape(Nx, Ny)

    # enforce BC in u itself
    u[0,:] = 0.0
    u[-1,:] = 0.0
    u[:,0] = 0.0
    u[:,-1] = 0.0

    # central differences with periodic-like rolls, then zero at boundary
    u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2

    # zero out second derivatives at boundaries to respect Dirichlet BC
    u_xx[0,:] = 0.0
    u_xx[-1,:] = 0.0
    u_xx[:,0] = 0.0
    u_xx[:,-1] = 0.0
    u_yy[0,:] = 0.0
    u_yy[-1,:] = 0.0
    u_yy[:,0] = 0.0
    u_yy[:,-1] = 0.0

    du_dt = eps*(u_xx + u_yy) - u**3 + u

    # enforce BC on du/dt at boundaries
    du_dt[0,:] = 0.0
    du_dt[-1,:] = 0.0
    du_dt[:,0] = 0.0
    du_dt[:,-1] = 0.0

    return du_dt.reshape(-1)

Nxg, Nyg, Ntg = Nx, Ny, Nt
xg = x.cpu().numpy()
yg = y.cpu().numpy()
tg = t.cpu().numpy()
dxg = float(dx.cpu().numpy())
dyg = float(dy.cpu().numpy())

Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
u0 = 0.05*np.sin(np.pi*Xg)*np.sin(np.pi*Yg)
u0[0,:] = 0.0
u0[-1,:] = 0.0
u0[:,0] = 0.0
u0[:,-1] = 0.0

u0_flat = u0.reshape(-1)
utrue = odeint(allen_cahn_rhs, u0_flat, tg, args=(eps, Nxg, Nyg, dxg, dyg))
utrue = utrue.reshape(Ntg, Nxg, Nyg)  # [Nt, Nx, Ny]

# ======================================================
# 9. Evaluation with Trans-PINN
# ======================================================
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(Ntg):
        tt = torch.full((Nxg*Nyg, 1), tg[i], device=device)
        X_eval, Y_eval = torch.meshgrid(
            torch.tensor(xg, device=device).float(),
            torch.tensor(yg, device=device).float(),
            indexing="ij"
        )
        xx = X_eval.reshape(-1,1)
        yy = Y_eval.reshape(-1,1)
        inp = torch.cat([xx, yy, tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().reshape(Nxg, Nyg)

num_l1 = np.sum(np.abs(utrue - upred))
den_l1 = np.sum(np.abs(utrue)) + 1e-12
l1 = num_l1 / den_l1

num_l2 = np.sum((utrue - upred)**2)
den_l2 = np.sum(utrue**2) + 1e-12
l2 = np.sqrt(num_l2 / den_l2)

print("\nFINAL TRANS-PINN (2D Allen–Cahn)")
print(f"Relative L1 Error: {l1:.6e}")
print(f"Relative L2 Error: {l2:.6e}")

# ======================================================
# 10. Plotting at final time t=1
# ======================================================
idx_final = -1  # last time step

X_plot, Y_plot = np.meshgrid(xg, yg, indexing="ij")

fig = plt.figure(figsize=(18,5))

# -------- FDM Exact (3D) --------
ax1 = fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(X_plot, Y_plot, utrue[idx_final], cmap='viridis')
ax1.set_title("FDM Ground Truth")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# -------- Trans-PINN Prediction (3D) --------
ax2 = fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(X_plot, Y_plot, upred[idx_final], cmap='viridis')
ax2.set_title("General Trans-PINN")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u")
ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# -------- Absolute Error (3D) --------
ax3 = fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(X_plot, Y_plot,
                 np.abs(utrue[idx_final] - upred[idx_final]),
                 cmap='hot')
ax3.set_title("Absolute Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("|Error|")
ax3.grid(False)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig("TransPINN_AllenCahn_3D.png", dpi=600)
plt.show()
