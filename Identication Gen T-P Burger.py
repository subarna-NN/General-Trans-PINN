import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from scipy.integrate import odeint
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

nu = 0.01 / np.pi
pi = torch.tensor(np.pi, device=device)
print(f"nu     : {nu:.6f}")

# ------------------------------------------------------------------
# Pseudo-sequence generator  (ORIGINAL — unchanged)
# ------------------------------------------------------------------
def pseudo_sequence(xt, dx):
    """Input: xt=[N,2]  Output: [N,3,2]  (left, center, right neighbors)"""
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.stack([
        torch.cat([x - dx, t], dim=1),
        torch.cat([x,       t], dim=1),
        torch.cat([x + dx, t], dim=1)
    ], dim=1)

# ------------------------------------------------------------------
# Transformer block  (ORIGINAL — unchanged, Tanh feed-forward)
# ------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, 256), nn.Tanh(),
            nn.Linear(256, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x    = self.norm1(x + h)
        return self.norm2(x + self.ff(x))

# ------------------------------------------------------------------
# General Trans-PINN  (ORIGINAL — unchanged)
# ------------------------------------------------------------------
class TransPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed       = nn.Linear(2, 64)
        self.transformer = TransformerBlock(64, 4)
        self.mlp         = nn.Sequential(
            nn.Linear(64,  128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, xt, dx):
        seq = pseudo_sequence(xt, dx)   # [N, 3, 2]
        h   = self.embed(seq)           # [N, 3, 64]
        h   = self.transformer(h)       # [N, 3, 64]
        return self.mlp(h[:, 1, :])     # center token → [N, 1]

model = TransPINN().to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------
# Training data  (EQUALIZED)
# ------------------------------------------------------------------
N_RES = 10_000
dx    = torch.tensor(2.0 / 200, device=device)   # spatial step used in pseudo-seq

# Collocation points — random sampling, fixed throughout training
def new_colloc():
    pts = torch.rand(N_RES, 2, device=device)
    pts[:, 0] = pts[:, 0] * 2.0 - 1.0   # x: [-1, 1]
    return pts

res = new_colloc().requires_grad_(True)

# IC  (t = 0)
x_ic = torch.linspace(-1, 1, 512).reshape(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic)
ic   = torch.cat([x_ic, t_ic], dim=1)
u_ic_true = -torch.sin(pi * x_ic)

# BC  (x = ±1, all t)
t_bc = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
bc_l = torch.cat([-torch.ones_like(t_bc), t_bc], dim=1)
bc_r = torch.cat([ torch.ones_like(t_bc), t_bc], dim=1)

# ------------------------------------------------------------------
# Loss function  (EQUALIZED — all weights = 1.0)
# ------------------------------------------------------------------
def pinn_loss():
    u    = model(res, dx)
    g1   = torch.autograd.grad(u.sum(), res, create_graph=True)[0]
    u_x  = g1[:, 0:1]
    u_t  = g1[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), res, create_graph=True)[0][:, 0:1]

    f        = u_t + u * u_x - nu * u_xx
    loss_res = torch.mean(f ** 2)
    loss_ic  = torch.mean((model(ic, dx) - u_ic_true) ** 2)
    loss_bc  = (torch.mean(model(bc_l, dx) ** 2) +
                torch.mean(model(bc_r, dx) ** 2))

    # Equal weights = 1.0 for all terms
    return loss_res + loss_ic + loss_bc

# ------------------------------------------------------------------
# Training  (EQUALIZED)
# ------------------------------------------------------------------
start_time = time.time()

# Stage 1 : Adam — 10,000 steps, lr=1e-3, no scheduler
print("\nStage 1/2 — Adam (10,000 steps) ...")
opt = Adam(model.parameters(), lr=1e-3)

for step in tqdm(range(10_000)):
    opt.zero_grad()
    loss = pinn_loss()
    loss.backward()
    opt.step()
    if (step + 1) % 2000 == 0:
        tqdm.write(f"  [{step+1:5d}] loss = {loss.item():.4e}")

# Stage 2 : L-BFGS — max_iter=5000, strong_wolfe
print("\nStage 2/2 — L-BFGS (max_iter=5000) ...")
opt_lbfgs = LBFGS(
    model.parameters(), max_iter=5000,
    line_search_fn="strong_wolfe"
)

def closure():
    opt_lbfgs.zero_grad()
    loss = pinn_loss()
    loss.backward()
    return loss

opt_lbfgs.step(closure)
print("  L-BFGS done.")

total_time = time.time() - start_time
print(f"\nTotal training time : {total_time:.1f} s")

# ------------------------------------------------------------------
# FDM ground truth
# ------------------------------------------------------------------
def get_fdm_truth(nu_val, nx=512, nt=200):
    xg  = np.linspace(-1, 1, nx)
    dxg = xg[1] - xg[0]
    tg  = np.linspace(0, 1, nt)
    u0  = -np.sin(np.pi * xg[1:-1])

    def rhs(u, t_val):
        uf   = np.concatenate(([0.0], u, [0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dxg**2
        u_x  = (uf[2:] - uf[:-2]) / (2 * dxg)
        return -uf[1:-1] * u_x + nu_val * u_xx

    sol = odeint(rhs, u0, tg)
    u   = np.zeros((nt, nx))
    u[:, 1:-1] = sol
    return xg, tg, u

xg, tg, utrue = get_fdm_truth(nu)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(len(tg)):
        tt  = torch.full((len(xg), 1), tg[i], device=device)
        xx  = torch.tensor(xg, dtype=torch.float32,
                           device=device).reshape(-1, 1)
        inp = torch.cat([xx, tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().flatten()

rl1 = np.sum(np.abs(utrue - upred)) / (np.sum(np.abs(utrue)) + 1e-12)
rl2 = np.sqrt(np.sum((utrue - upred)**2) /
              (np.sum(utrue**2) + 1e-12))

print(f"\n{'='*50}")
print(f"  General Trans-PINN  —  FAIR COMPARISON RESULT")
print(f"  Relative L1 Error : {rl1:.6f}  ({rl1*100:.4f}%)")
print(f"  Relative L2 Error : {rl2:.6f}  ({rl2*100:.4f}%)")
print(f"{'='*50}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kw = dict(aspect='auto', extent=[0, 1, -1, 1], origin='lower', cmap='turbo')

im0 = axes[0].imshow(utrue, **kw)
axes[0].set_title("FDM Ground Truth"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(upred, **kw)
axes[1].set_title("General Trans-PINN (Fair)"); axes[1].set_xlabel("t")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(utrue - upred),
                     aspect='auto', extent=[0, 1, -1, 1],
                     origin='lower', cmap='inferno')
axes[2].set_title("Absolute Error"); axes[2].set_xlabel("t")
plt.colorbar(im2, ax=axes[2], label="|Error|")

plt.suptitle(f"", fontsize=11)
plt.tight_layout()
plt.savefig("fair_General_TransPINN_result.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: fair_General_TransPINN_result.png")
