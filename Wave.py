import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm

# ======================================================
# 1. Device & Parameters
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

print("Using device:", device)

# ======================================================
# 2. Pseudo-sequence generator
# ======================================================
def pseudo_sequence(xt, dx):
    x = xt[:,0:1]
    y = xt[:,1:2]
    t = xt[:,2:3]

    return torch.stack([
        torch.cat([x-dx, y, t], dim=1),
        torch.cat([x,    y, t], dim=1),
        torch.cat([x+dx, y, t], dim=1)
    ], dim=1)

# ======================================================
# 3. Transformer Block
# ======================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model,256),
            nn.Tanh(),
            nn.Linear(256,d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        h,_ = self.attn(x,x,x)
        x = self.norm1(x+h)
        h = self.ff(x)
        return self.norm2(x+h)

# ======================================================
# 4. Trans-PINN Model
# ======================================================
class TransPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3,64)
        self.transformer = TransformerBlock(64,4)
        self.mlp = nn.Sequential(
            nn.Linear(64,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self,xt,dx):
        seq = pseudo_sequence(xt,dx)
        h = self.embed(seq)
        h = self.transformer(h)
        h = h[:,1,:]
        return self.mlp(h)

model = TransPINN().to(device)
print("Total parameters:",
      sum(p.numel() for p in model.parameters()))
# ======================================================
# 5. Domain Grid
# ======================================================
Nx, Ny, Nt = 41, 41, 120

x = torch.linspace(0,1,Nx)
y = torch.linspace(0,1,Ny)
t = torch.linspace(0,1,Nt)

dx = x[1]-x[0]
dy = y[1]-y[0]
dt = t[1]-t[0]

N_res = 35000   # important

x_res = torch.rand(N_res,1)
y_res = torch.rand(N_res,1)
t_res = torch.rand(N_res,1)

res = torch.cat([x_res,y_res,t_res],dim=1).to(device).requires_grad_(True)
# ======================================================
# 6. Initial & Boundary Conditions
# ======================================================
X_ic,Y_ic = torch.meshgrid(x,y,indexing="ij")
x_ic = X_ic.reshape(-1,1).to(device)
y_ic = Y_ic.reshape(-1,1).to(device)
t_ic = torch.zeros_like(x_ic)
ic = torch.cat([x_ic,y_ic,t_ic],dim=1).requires_grad_(True)

u_ic = torch.sin(np.pi*x_ic)*torch.sin(np.pi*y_ic)

# Boundary points
Nt_bc = 60
t_bc = torch.linspace(0,1,Nt_bc).to(device)

def boundary_points():
    pts = []
    y_vals = torch.linspace(0,1,Ny).to(device)
    x_vals = torch.linspace(0,1,Nx).to(device)

    Yb,Tb = torch.meshgrid(y_vals,t_bc,indexing="ij")
    X0 = torch.zeros_like(Yb)
    X1 = torch.ones_like(Yb)

    pts.append(torch.cat([X0.reshape(-1,1),Yb.reshape(-1,1),Tb.reshape(-1,1)],dim=1))
    pts.append(torch.cat([X1.reshape(-1,1),Yb.reshape(-1,1),Tb.reshape(-1,1)],dim=1))

    Xb,Tb = torch.meshgrid(x_vals,t_bc,indexing="ij")
    Y0 = torch.zeros_like(Xb)
    Y1 = torch.ones_like(Xb)

    pts.append(torch.cat([Xb.reshape(-1,1),Y0.reshape(-1,1),Tb.reshape(-1,1)],dim=1))
    pts.append(torch.cat([Xb.reshape(-1,1),Y1.reshape(-1,1),Tb.reshape(-1,1)],dim=1))

    return torch.cat(pts,dim=0)

bc = boundary_points().to(device)

# ======================================================
# 7. PINN Loss (Wave Equation)
# ======================================================
def pinn_loss():

    u = model(res,dx)

    grads = torch.autograd.grad(u.sum(),res,create_graph=True)[0]
    u_x = grads[:,0:1]
    u_y = grads[:,1:2]
    u_t = grads[:,2:3]

    grads_x = torch.autograd.grad(u_x.sum(),res,create_graph=True)[0]
    u_xx = grads_x[:,0:1]

    grads_y = torch.autograd.grad(u_y.sum(),res,create_graph=True)[0]
    u_yy = grads_y[:,1:2]

    grads_t = torch.autograd.grad(u_t.sum(),res,create_graph=True)[0]
    u_tt = grads_t[:,2:3]

    f = u_tt - (u_xx + u_yy)
    loss_res = torch.mean(f**2)

    # IC displacement
    u_ic_pred = model(ic,dx)
    loss_ic_disp = torch.mean((u_ic_pred-u_ic)**2)

    # IC velocity
    grads_ic = torch.autograd.grad(u_ic_pred.sum(),ic,create_graph=True)[0]
    u_t_ic = grads_ic[:,2:3]
    loss_ic_vel = torch.mean(u_t_ic**2)

    # BC
    u_bc = model(bc,dx)
    loss_bc = torch.mean(u_bc**2)

    return (
    1.0 * loss_res +
    10.0 * loss_ic_disp +
    10.0 * loss_ic_vel +
    5.0 * loss_bc
)

# ======================================================
# 8. Training
# ======================================================
print("Training...")
opt = Adam(model.parameters(),lr=1e-3)

for _ in tqdm(range(8000)):
    opt.zero_grad()
    loss = pinn_loss()
    loss.backward()
    opt.step()

print("L-BFGS polishing...")
opt_lbfgs = LBFGS(model.parameters(),max_iter=1200,lr=1.0,line_search_fn="strong_wolfe")

def closure():
    opt_lbfgs.zero_grad()
    l = pinn_loss()
    l.backward()
    return l

opt_lbfgs.step(closure)

# ======================================================
# 9. FDM Ground Truth (Central Difference)
# ======================================================
xg = x.numpy()
yg = y.numpy()
tg = t.numpy()

dx_val = float(dx.cpu().numpy())
dy_val = float(dy.cpu().numpy())
dt_val = float(dt.cpu().numpy())

u = np.zeros((Nt, Nx, Ny))

# Initial condition
Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
u[0] = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)

# --------------------------------------------------
# First time step (Taylor expansion, u_t = 0)
# --------------------------------------------------
u[1] = u[0].copy()

uxx = np.zeros_like(u[0])
uyy = np.zeros_like(u[0])

uxx[1:-1,1:-1] = (
    u[0,2:,1:-1] - 2*u[0,1:-1,1:-1] + u[0,:-2,1:-1]
) / dx_val**2

uyy[1:-1,1:-1] = (
    u[0,1:-1,2:] - 2*u[0,1:-1,1:-1] + u[0,1:-1,:-2]
) / dy_val**2

u[1,1:-1,1:-1] += 0.5 * dt_val**2 * (
    uxx[1:-1,1:-1] + uyy[1:-1,1:-1]
)

# --------------------------------------------------
# Time stepping (central difference in time)
# --------------------------------------------------
for n in range(1, Nt-1):

    uxx = np.zeros_like(u[n])
    uyy = np.zeros_like(u[n])

    uxx[1:-1,1:-1] = (
        u[n,2:,1:-1] - 2*u[n,1:-1,1:-1] + u[n,:-2,1:-1]
    ) / dx_val**2

    uyy[1:-1,1:-1] = (
        u[n,1:-1,2:] - 2*u[n,1:-1,1:-1] + u[n,1:-1,:-2]
    ) / dy_val**2

    u[n+1,1:-1,1:-1] = (
        2*u[n,1:-1,1:-1]
        - u[n-1,1:-1,1:-1]
        + dt_val**2 * (uxx[1:-1,1:-1] + uyy[1:-1,1:-1])
    )

# Dirichlet boundaries automatically remain zero
utrue = u
# ======================================================
# 10. Evaluation
# ======================================================
upred = np.zeros_like(utrue)

model.eval()
with torch.no_grad():
    for i in range(Nt):
        tt = torch.full((Nx*Ny,1),tg[i],device=device)
        X_eval,Y_eval = torch.meshgrid(
            torch.tensor(xg,device=device),
            torch.tensor(yg,device=device),
            indexing="ij"
        )
        xx = X_eval.reshape(-1,1)
        yy = Y_eval.reshape(-1,1)
        inp = torch.cat([xx,yy,tt],dim=1)
        upred[i] = model(inp,dx).cpu().numpy().reshape(Nx,Ny)

l2 = np.sqrt(np.sum((utrue-upred)**2)/np.sum(utrue**2))
print("Relative L2 Error:",l2)
l1 = np.sum(np.abs(utrue - upred)) / np.sum(np.abs(utrue))
print("Relative L1 Error:", l1)
# ======================================================
# 11. Plot Final Time
# ======================================================
idx = -1
X_plot,Y_plot = np.meshgrid(xg,yg,indexing="ij")

fig = plt.figure(figsize=(18,5))

ax1 = fig.add_subplot(131,projection='3d')
ax1.plot_surface(X_plot,Y_plot,utrue[idx],cmap='viridis')
ax1.set_title("FDM Ground Truth")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

ax2 = fig.add_subplot(132,projection='3d')
ax2.plot_surface(X_plot,Y_plot,upred[idx],cmap='viridis')
ax2.set_title("General Trans-PINN")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u")
ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

ax3 = fig.add_subplot(133,projection='3d')
ax3.plot_surface(X_plot,Y_plot,np.abs(utrue[idx]-upred[idx]),cmap='hot')
ax3.set_title("Absolute Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("|Error|")
ax3.grid(False)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig("Trans_PINN_2D_Wave.png", dpi=600)
plt.show()
