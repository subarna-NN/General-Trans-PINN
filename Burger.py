import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from scipy.integrate import odeint
import time
torch.cuda.synchronize()
# ======================================================
# 1. Device & parameters
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

nu = 0.01 / np.pi
pi = torch.tensor(np.pi, device=device)

print(f"Using device: {device}")
print(f"nu = {nu:.6f}")

# ======================================================
# 2. Pseudo-sequence generator (SAFE)
# ======================================================
def pseudo_sequence(xt, dx):
    """
    Input:  xt = [N,2]  -> (x,t)
    Output: [N,3,2]     -> [(x-dx,t),(x,t),(x+dx,t)]
    """
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.stack([
        torch.cat([x - dx, t], dim=1),
        torch.cat([x,       t], dim=1),
        torch.cat([x + dx, t], dim=1)
    ], dim=1)

# ======================================================
# 3. Transformer block (NO dropout)
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
# 4. Trans-PINN with pseudo-sequence
# ======================================================
class TransPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(2, 64)
        self.transformer = TransformerBlock(64, 4)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, xt, dx):
        seq = pseudo_sequence(xt, dx)     # [N,3,2]
        h = self.embed(seq)               # [N,3,64]
        h = self.transformer(h)           # [N,3,64]
        h = h[:, 1, :]                    # center token
        return self.mlp(h)

model = TransPINN().to(device)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

# ======================================================
# 5. Data (GRID-based, stable)
# ======================================================
Nx, Nt = 200, 120
x = torch.linspace(-1, 1, Nx)
t = torch.linspace(0, 1, Nt)
X, T = torch.meshgrid(x, t, indexing="ij")
res = torch.cat([X.reshape(-1,1), T.reshape(-1,1)], dim=1)
res = res.to(device).requires_grad_(True)

dx = x[1] - x[0]

# IC
x_ic = torch.linspace(-1, 1, 512).reshape(-1,1).to(device)
t_ic = torch.zeros_like(x_ic)
ic = torch.cat([x_ic, t_ic], dim=1)
u_ic = -torch.sin(pi * x_ic)

# BC
t_bc = torch.linspace(0, 1, 300).reshape(-1,1).to(device)
x_l = -torch.ones_like(t_bc)
x_r =  torch.ones_like(t_bc)
bc_l = torch.cat([x_l, t_bc], dim=1)
bc_r = torch.cat([x_r, t_bc], dim=1)

# ======================================================
# 6. Loss function (same as working code)
# ======================================================
def pinn_loss():
    u = model(res, dx)

    grads = torch.autograd.grad(u.sum(), res, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), res, create_graph=True)[0][:, 0:1]

    f = u_t + u * u_x - nu * u_xx
    loss_res = torch.mean(f**2)

    loss_ic = torch.mean((model(ic, dx) - u_ic)**2)
    loss_bc = (
        torch.mean(model(bc_l, dx)**2) +
        torch.mean(model(bc_r, dx)**2)
    )

    return loss_res + loss_ic + loss_bc
start_time = time.time()
# ======================================================
# 7. Training
# ======================================================
print("Training Trans-PINN (Adam)...")
opt = Adam(model.parameters(), lr=1e-3)

for _ in tqdm(range(10000)):
    opt.zero_grad()
    loss = pinn_loss()
    loss.backward()
    opt.step()

print("Polishing with L-BFGS...")
opt_lbfgs = LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=800,
    line_search_fn="strong_wolfe"
)

def closure():
    opt_lbfgs.zero_grad()
    l = pinn_loss()
    l.backward()
    return l

opt_lbfgs.step(closure)

# ======================================================
# 8. FDM ground truth
# ======================================================
def get_fdm_truth(nu, nx=512, nt=200):
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]
    t = np.linspace(0, 1, nt)
    u0 = -np.sin(np.pi * x[1:-1])

    def rhs(u, t):
        uf = np.concatenate(([0.0], u, [0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dx**2
        u_x = (uf[2:] - uf[:-2]) / (2*dx)
        return -uf[1:-1]*u_x + nu*u_xx

    sol = odeint(rhs, u0, t)
    u = np.zeros((nt, nx))
    u[:,1:-1] = sol
    return x, t, u

xg, tg, utrue = get_fdm_truth(nu)

# ======================================================
# 9. Evaluation
# ======================================================
upred = np.zeros_like(utrue)
model.eval()

with torch.no_grad():
    for i in range(len(tg)):
        tt = torch.full((len(xg),1), tg[i], device=device)
        xx = torch.tensor(xg, device=device).reshape(-1,1).float()
        upred[i] = model(torch.cat([xx, tt], 1), dx).cpu().numpy().flatten()

l1 = np.sum(np.abs(utrue - upred)) / np.sum(np.abs(utrue))
l2 = np.sqrt(np.sum((utrue - upred)**2) / np.sum(utrue**2))

print("\nFINAL TRANS-PINN (WITH PSEUDO-SEQUENCE)")
print(f"Relative L1 Error: {l1:.6f}")
print(f"Relative L2 Error: {l2:.6f}")

# ======================================================
# 10. Plotting
# ======================================================
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(utrue, aspect='auto', extent=[0,1,-1,1], origin='lower', cmap='RdBu_r')
plt.colorbar(); plt.title("FDM Ground Truth"); plt.xlabel("t"); plt.ylabel("x")

plt.subplot(1,3,2)
plt.imshow(upred, aspect='auto', extent=[0,1,-1,1], origin='lower', cmap='RdBu_r')
plt.colorbar(); plt.title("General Trans-PINN"); plt.xlabel("t")

plt.subplot(1,3,3)
plt.imshow(np.abs(utrue-upred), aspect='auto', extent=[0,1,-1,1], origin='lower', cmap='Reds')
plt.colorbar(label="|Error|"); plt.title("Absolute Error"); plt.xlabel("t")

plt.tight_layout()
plt.savefig("TransPINN_Final.png", dpi=600)
plt.show()
print("Total Time:", time.time() - start_time, "seconds")
