"""
RD OOD-grid experiment: train CNN, MaskedCNN, UNet-1L, DB-Fast on a selected
training grid N (with N < 16), evaluate on [N, round(1.2*N), round(1.6*N),
2*N] by default (all <= 32 by construction), and save raw results +
training logs to JSON for later plotting and analysis.

Usage:
    python experiment.py                        # default run_dir = run_default
    python experiment.py --run-dir run_seed42   # custom dir
    python experiment.py --epochs 50 --db-epochs 100  # quick test
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


# ---------- problem setup ----------
D, SIGMA = 0.01, 1.0


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rd_matrix(N, D=0.01, sigma=1.0):
    h = 1. / (N + 1); n = N * N
    d0 = -4. * np.ones(n); d1 = np.ones(n - 1); dN = np.ones(n - N)
    for i in range(1, N):
        d1[i * N - 1] = 0.
    L = diags([d0, d1, d1, dN, dN], [0, 1, -1, N, -N], format='csr') / (h ** 2)
    return -D * L + sigma * eye(n, format='csr')


def solve_rd(f_flat, N, D=0.01, sigma=1.0):
    return spsolve(build_rd_matrix(N, D, sigma), f_flat)


def random_source_term(N, n_modes=4):
    h = 1. / (N + 1)
    xs = np.arange(1, N + 1) * h; ys = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(xs, ys); f = np.zeros((N, N))
    for _ in range(n_modes):
        cx = np.random.uniform(0.1, 0.9); cy = np.random.uniform(0.1, 0.9)
        amp = np.random.randn(); w = np.random.uniform(0.05, 0.2)
        f += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2))
    return f


def make_rd_dataset(num_samples, N, D=0.01, sigma=1.0):
    h = 1. / (N + 1)
    xs = np.arange(1, N + 1) * h; ys = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(xs, ys)
    coords_t = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=-1), dtype=torch.float32)
    ds = []
    for _ in range(num_samples):
        f = random_source_term(N); u = solve_rd(f.flatten(), N, D, sigma)
        u_std = max(u.std(), 1e-8)
        ds.append((
            torch.tensor(f.flatten() / u_std, dtype=torch.float32),
            torch.tensor(u / u_std, dtype=torch.float32),
            coords_t))
    return ds


def stack_dataset_on_device(ds, device):
    """Stack a list-of-tuples dataset into pre-stacked tensors on `device`.
    Returns (f_all, u_all, coords) where:
        f_all : (M, N*N) float32
        u_all : (M, N*N) float32
        coords: (N*N, 2) float32  (shared across all samples)
    Doing this once eliminates per-batch list comp + stack + .to(device).
    """
    f_all = torch.stack([s[0] for s in ds]).to(device, non_blocking=True)
    u_all = torch.stack([s[1] for s in ds]).to(device, non_blocking=True)
    coords = ds[0][2].to(device, non_blocking=True)
    return f_all, u_all, coords


# ---------- models ----------
def group_norm(num_channels):
    """GroupNorm with the largest group count <=8 that divides num_channels.
    This allows parameter matching with arbitrary channel widths, not only
    widths divisible by 8.
    """
    for g in range(min(8, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def coords_to_2d(coords, N, B, dev):
    return (coords.to(dev)
            .reshape(N, N, 2).permute(2, 0, 1)
            .unsqueeze(0).expand(B, -1, -1, -1))


class CNNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), group_norm(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), group_norm(ch))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class CNNSolver(nn.Module):
    def __init__(self, d_model=64, num_layers=8):
        super().__init__()
        self.in_proj = nn.Conv2d(3, d_model, 1)
        self.layers = nn.ModuleList([CNNBlock(d_model) for _ in range(num_layers)])
        self.out_proj = nn.Conv2d(d_model, 1, 1)

    def forward(self, f, coords, N):
        B = f.shape[0]; c2d = coords_to_2d(coords, N, B, f.device)
        x = self.in_proj(torch.cat([f.reshape(B, 1, N, N), c2d], dim=1))
        for l in self.layers:
            x = l(x)
        return self.out_proj(x).reshape(B, N * N)


class MaskedCNNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch, 1, 3, 3) * 0.125)
        mask = torch.ones(1, 1, 3, 3); mask[0, 0, 1, 1] = 0.
        self.register_buffer('mask', mask)
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch * 4, 1), nn.GELU(), nn.Conv2d(ch * 4, ch, 1))
        self.n1 = group_norm(ch); self.n2 = group_norm(ch)

    def forward(self, x):
        msg = F.conv2d(x, self.w * self.mask, padding=1, groups=x.shape[1])
        h = self.n1(x + msg)
        return self.n2(h + self.mlp(h))


class MaskedCNNSolver(nn.Module):
    def __init__(self, d_model=64, num_layers=8):
        super().__init__()
        self.in_proj = nn.Conv2d(3, d_model, 1)
        self.layers = nn.ModuleList([MaskedCNNBlock(d_model) for _ in range(num_layers)])
        self.out_proj = nn.Conv2d(d_model, 1, 1)

    def forward(self, f, coords, N):
        B = f.shape[0]; c2d = coords_to_2d(coords, N, B, f.device)
        x = self.in_proj(torch.cat([f.reshape(B, 1, N, N), c2d], dim=1))
        for l in self.layers:
            x = l(x)
        return self.out_proj(x).reshape(B, N * N)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            group_norm(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            group_norm(out_ch), nn.GELU())
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


def make_level(in_ch, out_ch, num_layers):
    blocks = [UNetBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(num_layers)]
    return nn.Sequential(*blocks)


class UNet1LSolver(nn.Module):
    def __init__(self, d_model=64, num_layers=3):
        super().__init__()
        d = d_model
        self.enc1 = make_level(3, d, num_layers)
        self.enc2 = make_level(d, d * 2, num_layers)
        self.down = nn.AvgPool2d(2, ceil_mode=True)
        self.dec1 = make_level(d * 3, d, num_layers)
        self.out = nn.Conv2d(d, 1, 1)

    def forward(self, f, coords, N):
        B = f.shape[0]; c2d = coords_to_2d(coords, N, B, f.device)
        x = torch.cat([f.reshape(B, 1, N, N), c2d], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        up = F.interpolate(e2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up, e1], dim=1))
        return self.out(d1).reshape(B, N * N)


class UNet2LSolver(nn.Module):
    def __init__(self, d_model=32, num_layers=2):
        super().__init__()
        d = d_model
        self.enc1 = make_level(3, d, num_layers)
        self.enc2 = make_level(d, d * 2, num_layers)
        self.enc3 = make_level(d * 2, d * 4, num_layers)
        self.down = nn.AvgPool2d(2, ceil_mode=True)
        self.dec2 = make_level(d * 6, d * 2, num_layers)
        self.dec1 = make_level(d * 3, d, num_layers)
        self.out = nn.Conv2d(d, 1, 1)

    def forward(self, f, coords, N):
        B = f.shape[0]; c2d = coords_to_2d(coords, N, B, f.device)
        x = torch.cat([f.reshape(B, 1, N, N), c2d], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        up2 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        return self.out(d1).reshape(B, N * N)


class DBEdgeConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d, 1, 3, 3) * 0.25)
        mask = torch.zeros(1, 1, 3, 3)
        mask[0, 0, 0, 1] = mask[0, 0, 2, 1] = mask[0, 0, 1, 0] = mask[0, 0, 1, 2] = 1.
        self.register_buffer('mask', mask)
        self.norm = group_norm(d)

    def forward(self, x):
        return self.norm(x + F.conv2d(x, self.w * self.mask, padding=1, groups=x.shape[1]))


class DBTriConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d, 1, 3, 3) * 0.25)
        mask = torch.zeros(1, 1, 3, 3)
        mask[0, 0, 0, 0] = mask[0, 0, 0, 2] = mask[0, 0, 2, 0] = mask[0, 0, 2, 2] = 1.
        self.register_buffer('mask', mask)
        self.gate = nn.Conv2d(d, d, 1); self.norm = group_norm(d)

    def forward(self, x):
        cent = F.conv2d(x, self.w * self.mask, padding=1, groups=x.shape[1])
        return self.norm(x + torch.sigmoid(self.gate(x)) * (cent - x))


class DBLayerFast(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.edge = DBEdgeConv(d); self.tri = DBTriConv(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, d * 4, 1), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(d * 4, d, 1))
        self.norm = group_norm(d); self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.edge(x); x = self.tri(x)
        return self.norm(x + self.drop(self.mlp(x)))


class DBSolverFast(nn.Module):
    def __init__(self, d_model=64, num_layers=8, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv2d(3, d_model, 1)
        self.layers = nn.ModuleList([DBLayerFast(d_model, dropout) for _ in range(num_layers)])
        self.norm = group_norm(d_model)
        self.out_head = nn.Conv2d(d_model, 1, 1)

    def forward(self, f, coords, N):
        B = f.shape[0]; c2d = coords_to_2d(coords, N, B, f.device)
        x = self.in_proj(torch.cat([f.reshape(B, 1, N, N), c2d], dim=1))
        for l in self.layers:
            x = l(x)
        return self.out_head(self.norm(x)).reshape(B, N * N)


# ---------- param matching ----------
def search_d(ModelClass, target, num_layers, tol=0.03, step=1, hi=512):
    best_d, best_gap = step, float('inf')
    for d in range(step, hi + step, step):
        try:
            p = count_params(ModelClass(d_model=d, num_layers=num_layers))
        except Exception:
            continue
        gap = abs(p - target) / target
        if gap < best_gap:
            best_gap, best_d = gap, d
        if p > target * (1 + tol * 3):
            break
    fp = count_params(ModelClass(d_model=best_d, num_layers=num_layers))
    return best_d, fp, abs(fp - target) / target


# ---------- train / eval ----------
def relative_l2(pred, target):
    return (((pred - target) ** 2).sum(-1).sqrt() /
            (target ** 2).sum(-1).sqrt().clamp(min=1e-8)).mean().item()


def train_model(model, train_pack, val_pack, device, N_train, num_epochs=200,
                batch_size=32, lr=1e-3, name='', val_every=5):
    """train_pack/val_pack are tuples (f_all, u_all, coords) on `device`."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-5)

    f_tr, u_tr, coords = train_pack
    f_va, u_va, coords_va = val_pack
    M_tr = f_tr.shape[0]; M_va = f_va.shape[0]

    logs = {'epoch': [], 'train_loss': [], 'val_rel_l2': []}
    for epoch in range(1, num_epochs + 1):
        model.train()
        # Shuffle indices on device (same as random.shuffle of list)
        perm = torch.randperm(M_tr, device=device)
        tl, steps = 0., 0
        for i in range(0, M_tr, batch_size):
            idx = perm[i:i + batch_size]
            f_b = f_tr[idx]
            u_b = u_tr[idx]
            loss = F.mse_loss(model(f_b, coords, N_train), u_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item(); steps += 1
        sched.step()
        train_loss = tl / max(steps, 1)

        # Validate every val_every epochs (and at the final epoch)
        if epoch % val_every == 0 or epoch == 1 or epoch == num_epochs:
            model.eval(); ve = []
            with torch.no_grad():
                for i in range(0, M_va, batch_size):
                    f_b = f_va[i:i + batch_size]
                    u_b = u_va[i:i + batch_size]
                    ve.append(relative_l2(model(f_b, coords_va, N_train), u_b))
            val_rel = sum(ve) / len(ve)
            logs['epoch'].append(epoch)
            logs['train_loss'].append(train_loss)
            logs['val_rel_l2'].append(val_rel)
            if epoch % 40 == 0 or epoch == 1 or epoch == num_epochs:
                print(f'[{name}] Ep{epoch:3d} '
                      f'loss={train_loss:.4f} val={val_rel:.4f}')
    return model, logs


def evaluate(model, eval_pack, N, device, batch_size=32):
    """eval_pack is (f_all, u_all, coords) on `device`."""
    f_all, u_all, coords = eval_pack
    M = f_all.shape[0]
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, M, batch_size):
            f_b = f_all[i:i + batch_size]
            u_b = u_all[i:i + batch_size]
            errs.append(relative_l2(model(f_b, coords, N), u_b))
    return sum(errs) / len(errs)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', '--run_dir', '--out_dir', dest='run_dir', default='run_default')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='list of random seeds; one full train+eval per seed, '
                         'results averaged. Default: 0 (Stage 1 single-seed)')
    ap.add_argument('--n-train', type=int, default=12,
                    help='training grid size N, must satisfy 8 <= N < 16 so '
                         'that 2*N <= 30 < 32 in the default eval scheme')
    ap.add_argument('--eval-grids', type=int, nargs='+', default=None,
                    help='grid sizes to evaluate on. If omitted, uses '
                         '[N, round(1.2*N), round(1.6*N), 2*N] (e.g. N=10 -> '
                         '10,12,16,20; N=12 -> 12,14,19,24; N=14 -> '
                         '14,17,22,28; N=15 -> 15,18,24,30). All grids must '
                         'be <= 32.')
    ap.add_argument('--epochs', type=int, default=200,
                    help='epochs for CNN/MaskedCNN/UNet-1L (Stage 1/2 default)')
    ap.add_argument('--db-epochs', type=int, default=300,
                    help='epochs for DB-Fast, typically 3x --epochs '
                         '(Stage 1/2 default)')
    ap.add_argument('--n-train-samples', type=int, default=800)
    ap.add_argument('--n-val-samples', type=int, default=200)
    ap.add_argument('--n-eval-samples', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--val-every', type=int, default=5,
                    help='run validation every K epochs (default 5). Lower = '
                         'denser val curve at higher cost.')
    args = ap.parse_args()

    N_train = args.n_train
    if not (8 <= N_train < 16):
        raise SystemExit(f'--n-train must satisfy 8 <= N < 16, got {N_train}')

    if args.eval_grids is None:
        eval_grids = sorted({N_train,
                             int(round(1.2 * N_train)),
                             int(round(1.6 * N_train)),
                             2 * N_train})
        if len(eval_grids) != 4:
            raise SystemExit(
                f'Default eval-grid scheme produced non-distinct grids for '
                f'N_train={N_train}: {eval_grids}. Pass --eval-grids '
                f'explicitly.')
    else:
        eval_grids = sorted(set(args.eval_grids))
        if N_train not in eval_grids:
            raise SystemExit(
                f'--n-train ({N_train}) must be in --eval-grids ({eval_grids})')

    if max(eval_grids) > 32:
        raise SystemExit(
            f'All eval grids must be <= 32 (1/N >= 0.03); got {eval_grids}.')

    seeds = list(args.seeds)
    print(f'N_train={N_train}, eval_grids={eval_grids}, seeds={seeds}')

    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Run dir: {args.run_dir}')

    # ----- data: shared across all seeds for fair comparison -----
    set_seed(0)
    print('Generating datasets (shared across seeds)...')
    train_ds = make_rd_dataset(args.n_train_samples, N_train)
    val_ds = make_rd_dataset(args.n_val_samples, N_train)
    eval_datasets = {N: make_rd_dataset(args.n_eval_samples, N) for N in eval_grids}
    print(f'Train={len(train_ds)}  Val={len(val_ds)}  Eval per grid={args.n_eval_samples}')

    # Move datasets to device once. This eliminates per-batch list comp +
    # stack + .to(device), which is ~all of the data-handling overhead.
    train_pack = stack_dataset_on_device(train_ds, device)
    val_pack = stack_dataset_on_device(val_ds, device)
    eval_packs = {N: stack_dataset_on_device(eval_datasets[N], device)
                  for N in eval_grids}
    # Free the Python-list versions; the GPU tensors are all we need.
    del train_ds, val_ds, eval_datasets

    # ----- param-matching search (deterministic, doesn't depend on seed) -----
    NUM_LAYERS_FLAT = 8
    NUM_LAYERS_UNET1 = 3

    cnn_ref = CNNSolver(d_model=64, num_layers=NUM_LAYERS_FLAT)
    TARGET = count_params(cnn_ref)
    print(f'Target params: {TARGET:,}')

    D_CNN = 64
    D_MASKED, _, _ = search_d(MaskedCNNSolver, TARGET, NUM_LAYERS_FLAT)
    D_U1, _, _ = search_d(UNet1LSolver, TARGET, NUM_LAYERS_UNET1)
    D_DB, _, _ = search_d(DBSolverFast, TARGET, NUM_LAYERS_FLAT)

    # Enforce the protocol's ±3% parameter-budget constraint.
    param_checks = {
        'CNN': count_params(CNNSolver(d_model=D_CNN, num_layers=NUM_LAYERS_FLAT)),
        'MaskedCNN': count_params(MaskedCNNSolver(d_model=D_MASKED, num_layers=NUM_LAYERS_FLAT)),
        'UNet-1L': count_params(UNet1LSolver(d_model=D_U1, num_layers=NUM_LAYERS_UNET1)),
        'DB-Fast': count_params(DBSolverFast(d_model=D_DB, num_layers=NUM_LAYERS_FLAT)),
    }
    bad = {n: abs(p - TARGET) / TARGET for n, p in param_checks.items()
           if abs(p - TARGET) / TARGET > 0.03}
    if bad:
        msg = ', '.join(f'{n}: gap={gap:.2%}' for n, gap in bad.items())
        raise SystemExit(f'Parameter matching failed protocol tolerance ±3%: {msg}')

    def build_models():
        return [
            ('CNN', CNNSolver(d_model=D_CNN, num_layers=NUM_LAYERS_FLAT), args.epochs),
            ('MaskedCNN', MaskedCNNSolver(d_model=D_MASKED, num_layers=NUM_LAYERS_FLAT), args.epochs),
            ('UNet-1L', UNet1LSolver(d_model=D_U1, num_layers=NUM_LAYERS_UNET1), args.epochs),
            ('DB-Fast', DBSolverFast(d_model=D_DB, num_layers=NUM_LAYERS_FLAT), args.db_epochs),
        ]

    model_names = [n for n, _, _ in build_models()]
    model_meta = {n: {'params': count_params(m), 'epochs': ep}
                  for n, m, ep in build_models()}
    for n in model_names:
        print(f'  {n:>10}: {model_meta[n]["params"]:>10,} params, '
              f'{model_meta[n]["epochs"]} epochs')

    # ----- per-seed loop -----
    # per_seed_results[seed][model_name][N] = rel_l2
    per_seed_results = {}
    per_seed_logs = {}
    t0 = time.time()
    for si, seed in enumerate(seeds):
        print(f'\n{"="*60}\nSEED {seed} ({si+1}/{len(seeds)})\n{"="*60}')
        set_seed(seed)
        models_info = build_models()

        seed_results = {}
        seed_logs = {}
        for name, m, ep in models_info:
            print(f'\n--- Seed {seed}, training {name} ({ep} epochs) ---')
            m, logs = train_model(m, train_pack, val_pack, device, N_train, ep,
                                  args.batch_size, 1e-3, name,
                                  val_every=args.val_every)
            seed_logs[name] = logs

            # eval on all grids
            seed_results[name] = {}
            for N in eval_grids:
                seed_results[name][N] = evaluate(m, eval_packs[N], N,
                                                 device, args.batch_size)
                print(f'  {name:>10}  N={N:3d}: rel-L2={seed_results[name][N]:.4f}')
            del m  # free GPU memory before next model

        per_seed_results[seed] = seed_results
        per_seed_logs[seed] = seed_logs

    # ----- aggregate across seeds -----
    Nmax = max(eval_grids)
    ood_grids = [N for N in eval_grids if N != N_train]

    def stack(name, N):
        """List of rel-L2 for this (model, N) across seeds."""
        return [per_seed_results[s][name][N] for s in seeds]

    # Per-model summaries with mean/std across seeds
    model_summaries = {}
    for name in model_names:
        # Per-seed scalar metrics, then aggregate
        id_per = [per_seed_results[s][name][N_train] for s in seeds]
        ood_avg_per = [
            float(np.mean([per_seed_results[s][name][N] for N in ood_grids]))
            for s in seeds
        ]
        worst_per = [per_seed_results[s][name][Nmax] for s in seeds]

        # log-log slope of rel_L2 vs N per seed (degradation rate; lower = flatter)
        Ns = np.array(eval_grids, dtype=float)
        slope_per = []
        for s in seeds:
            vals = np.array([per_seed_results[s][name][N] for N in eval_grids],
                            dtype=float)
            slope, _ = np.polyfit(np.log(Ns),
                                  np.log(np.maximum(vals, 1e-12)), 1)
            slope_per.append(float(slope))

        # Per-N mean/std curve
        rel_l2_mean = {str(N): float(np.mean(stack(name, N))) for N in eval_grids}
        rel_l2_std = {str(N): float(np.std(stack(name, N), ddof=0))
                      for N in eval_grids}

        model_summaries[name] = {
            'id_rel_l2_mean': float(np.mean(id_per)),
            'id_rel_l2_std': float(np.std(id_per, ddof=0)),
            'ood_avg_rel_l2_mean': float(np.mean(ood_avg_per)),
            'ood_avg_rel_l2_std': float(np.std(ood_avg_per, ddof=0)),
            'worst_ood_rel_l2_mean': float(np.mean(worst_per)),
            'worst_ood_rel_l2_std': float(np.std(worst_per, ddof=0)),
            'log_log_slope_mean': float(np.mean(slope_per)),
            'log_log_slope_std': float(np.std(slope_per, ddof=0)),
            'rel_l2_by_grid_mean': rel_l2_mean,
            'rel_l2_by_grid_std': rel_l2_std,
            'per_seed_ood_avg': ood_avg_per,
            'per_seed_log_log_slope': slope_per,
            'params': model_meta[name]['params'],
            'epochs': model_meta[name]['epochs'],
        }

    # ----- pairwise effects (DB-Fast as focal model) -----
    def gain_ood_avg(baseline, challenger):
        return (model_summaries[baseline]['ood_avg_rel_l2_mean'] -
                model_summaries[challenger]['ood_avg_rel_l2_mean'])

    def gain_slope(baseline, challenger):
        return (model_summaries[baseline]['log_log_slope_mean'] -
                model_summaries[challenger]['log_log_slope_mean'])

    effects_ood_avg = {
        'dbfast_vs_cnn': gain_ood_avg('CNN', 'DB-Fast'),
        'dbfast_vs_maskedcnn': gain_ood_avg('MaskedCNN', 'DB-Fast'),
        'dbfast_vs_unet1l': gain_ood_avg('UNet-1L', 'DB-Fast'),
    }
    effects_slope = {
        'dbfast_vs_cnn': gain_slope('CNN', 'DB-Fast'),
        'dbfast_vs_maskedcnn': gain_slope('MaskedCNN', 'DB-Fast'),
        'dbfast_vs_unet1l': gain_slope('UNet-1L', 'DB-Fast'),
    }
    effects_worst_ood = {
        f'dbfast_vs_{b.lower().replace("-","")}':
            model_summaries[b]['worst_ood_rel_l2_mean']
            - model_summaries['DB-Fast']['worst_ood_rel_l2_mean']
        for b in ['CNN', 'MaskedCNN', 'UNet-1L']
    }

    # ----- ranking summary across seeds -----
    # How often is DB-Fast best per seed (per metric)?
    def per_seed_winner(metric_fn):
        wins = {n: 0 for n in model_names}
        for s in seeds:
            vals = {n: metric_fn(s, n) for n in model_names}
            wins[min(vals, key=vals.get)] += 1
        return wins

    win_count_ood_avg = per_seed_winner(
        lambda s, n: float(np.mean(
            [per_seed_results[s][n][N] for N in ood_grids])))
    win_count_slope = per_seed_winner(
        lambda s, n: model_summaries[n]['per_seed_log_log_slope'][seeds.index(s)])
    win_count_worst = per_seed_winner(
        lambda s, n: per_seed_results[s][n][Nmax])

    config_dict = {
        'seeds': seeds,
        'N_train': N_train,
        'eval_grids': eval_grids,
        'D': D, 'sigma': SIGMA,
        'target_params': int(TARGET),
        'tolerance': 0.03,
        'n_train_samples': args.n_train_samples,
        'n_val_samples': args.n_val_samples,
        'n_eval_samples': args.n_eval_samples,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'db_epochs': args.db_epochs,
        'val_every': args.val_every,
        'device': str(device),
    }

    # ----- headline scalars in AI-Scientist format -----
    # AI-Scientist (launch_scientist.py do_idea) expects every value in
    # final_info.json to be a dict with at least a "means" key, e.g.
    #   baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    # We therefore wrap each metric as {"means": ..., "stds": ...}. For
    # quantities without a meaningful std (effects, win counts, num_seeds)
    # we set stds=0.0 so the format is uniform.
    focal = 'DB-Fast'

    def ms(mean, std=0.0):
        return {'means': float(mean), 'stds': float(std)}

    final_info = {
        # ---- focal-model (DB-Fast) headlines ----
        # PRIMARY metric first: log-log slope
        'best_log_log_slope': ms(model_summaries[focal]['log_log_slope_mean'],
                                 model_summaries[focal]['log_log_slope_std']),
        'best_ood_avg_rel_l2': ms(model_summaries[focal]['ood_avg_rel_l2_mean'],
                                  model_summaries[focal]['ood_avg_rel_l2_std']),
        'best_worst_ood_rel_l2': ms(model_summaries[focal]['worst_ood_rel_l2_mean'],
                                    model_summaries[focal]['worst_ood_rel_l2_std']),
        'best_id_rel_l2': ms(model_summaries[focal]['id_rel_l2_mean'],
                             model_summaries[focal]['id_rel_l2_std']),

        # ---- per-model log-log slope (PRIMARY metric, lower=flatter) ----
        'cnn_log_log_slope': ms(model_summaries['CNN']['log_log_slope_mean'],
                                model_summaries['CNN']['log_log_slope_std']),
        'maskedcnn_log_log_slope': ms(model_summaries['MaskedCNN']['log_log_slope_mean'],
                                      model_summaries['MaskedCNN']['log_log_slope_std']),
        'unet1l_log_log_slope': ms(model_summaries['UNet-1L']['log_log_slope_mean'],
                                   model_summaries['UNet-1L']['log_log_slope_std']),
        'dbfast_log_log_slope': ms(model_summaries['DB-Fast']['log_log_slope_mean'],
                                   model_summaries['DB-Fast']['log_log_slope_std']),

        # ---- per-model OOD avg (secondary) ----
        'cnn_ood_avg_rel_l2': ms(model_summaries['CNN']['ood_avg_rel_l2_mean'],
                                 model_summaries['CNN']['ood_avg_rel_l2_std']),
        'maskedcnn_ood_avg_rel_l2': ms(model_summaries['MaskedCNN']['ood_avg_rel_l2_mean'],
                                       model_summaries['MaskedCNN']['ood_avg_rel_l2_std']),
        'unet1l_ood_avg_rel_l2': ms(model_summaries['UNet-1L']['ood_avg_rel_l2_mean'],
                                    model_summaries['UNet-1L']['ood_avg_rel_l2_std']),
        'dbfast_ood_avg_rel_l2': ms(model_summaries['DB-Fast']['ood_avg_rel_l2_mean'],
                                    model_summaries['DB-Fast']['ood_avg_rel_l2_std']),

        # ---- core effects: DB-Fast vs each baseline (positive = DB-Fast better) ----
        # slope first (PRIMARY)
        'dbfast_vs_cnn_slope': ms(effects_slope['dbfast_vs_cnn']),
        'dbfast_vs_maskedcnn_slope': ms(effects_slope['dbfast_vs_maskedcnn']),
        'dbfast_vs_unet1l_slope': ms(effects_slope['dbfast_vs_unet1l']),
        # OOD avg (secondary)
        'dbfast_vs_cnn_ood_avg': ms(effects_ood_avg['dbfast_vs_cnn']),
        'dbfast_vs_maskedcnn_ood_avg': ms(effects_ood_avg['dbfast_vs_maskedcnn']),
        'dbfast_vs_unet1l_ood_avg': ms(effects_ood_avg['dbfast_vs_unet1l']),

        # ---- per-seed win counts (out of len(seeds)) ----
        'dbfast_wins_slope': ms(win_count_slope.get('DB-Fast', 0)),
        'dbfast_wins_ood_avg': ms(win_count_ood_avg.get('DB-Fast', 0)),
        'dbfast_wins_worst_ood': ms(win_count_worst.get('DB-Fast', 0)),
        'num_seeds': ms(len(seeds)),
    }

    # ----- rich record -----
    all_results = {
        'config': config_dict,
        'model_summaries': model_summaries,
        'per_seed_rel_l2': {
            str(s): {
                name: {str(N): float(per_seed_results[s][name][N])
                       for N in eval_grids}
                for name in model_names
            } for s in seeds
        },
        'training_logs': {
            str(s): per_seed_logs[s] for s in seeds
        },
        'effects_ood_avg_rel_l2': effects_ood_avg,
        'effects_log_log_slope': effects_slope,
        'effects_worst_ood_rel_l2': effects_worst_ood,
        'win_counts': {
            'ood_avg': win_count_ood_avg,
            'log_log_slope': win_count_slope,
            'worst_ood': win_count_worst,
        },
        'wall_time_sec': time.time() - t0,
    }

    fp1 = os.path.join(args.run_dir, 'final_info.json')
    fp2 = os.path.join(args.run_dir, 'all_results.json')
    with open(fp1, 'w') as f:
        json.dump(final_info, f, indent=2)
    with open(fp2, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved {fp1}')
    print(f'Saved {fp2}  ({time.time() - t0:.1f}s total)')

    # ----- console summary -----
    print(f'\n{"="*60}\nSUMMARY (mean ± std over {len(seeds)} seeds)\n'
          f'PRIMARY METRIC: log-log slope (↓ = flatter generalization curve)\n'
          f'{"="*60}')
    print(f'{"Model":<12} {"log-log slope ★":>20} {"OOD avg":>16} '
          f'{"Worst OOD":>16}')
    for n in model_names:
        m = model_summaries[n]
        print(f'{n:<12} '
              f'{m["log_log_slope_mean"]:+.3f} ± {m["log_log_slope_std"]:.3f}      '
              f'{m["ood_avg_rel_l2_mean"]:.4f} ± {m["ood_avg_rel_l2_std"]:.4f}  '
              f'{m["worst_ood_rel_l2_mean"]:.4f} ± {m["worst_ood_rel_l2_std"]:.4f}')
    print(f'\nDB-Fast wins (out of {len(seeds)} seeds):  '
          f'slope ★={win_count_slope.get("DB-Fast",0)}  '
          f'OOD avg={win_count_ood_avg.get("DB-Fast",0)}  '
          f'worst-OOD={win_count_worst.get("DB-Fast",0)}')


if __name__ == '__main__':
    main()
