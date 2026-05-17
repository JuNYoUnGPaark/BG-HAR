from __future__ import annotations

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Config
# =========================
SEED = 258
MODEL_PATH = "UCI-HAR.pt"
X_PATH = "X_test.npy"

NUM_RUN_SAMPLES = 100
REPEAT_PER_SAMPLE = 5
NUM_THREADS = 4

torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# =========================
# Model definition
# =========================
class MiniMambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2, conv_kernel: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden_dim,
            bias=True,
        )

        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u):
        B, T, H = u.shape

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b_t = torch.tanh(self.b_proj(u))
        c_t = torch.tanh(self.c_proj(u))

        A = torch.exp(self.A_log).view(1, H)

        state = torch.zeros(B, H, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(T):
            dt_cur = dt[:, t, :]
            u_cur = u[:, t, :]
            b_cur = b_t[:, t, :]
            c_cur = c_t[:, t, :]

            a_bar = torch.exp(-dt_cur * A)
            b_bar = (1.0 - a_bar) * b_cur

            state = a_bar * state + b_bar * u_cur
            y_cur = c_cur * state + self.D.view(1, H) * u_cur

            outputs.append(y_cur)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)

        y = self.selective_scan(u)
        y = y * F.silu(z)

        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class BenefitGatedSharedMambaHAR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        total_blocks: int = 3,
        early_exit_blocks: int = 1,
        dropout: float = 0.05,
        gate_hidden_dim: int = 32,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.total_blocks = total_blocks
        self.early_exit_blocks = early_exit_blocks

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            MiniMambaBlock(hidden_dim=hidden_dim, dropout=dropout, conv_kernel=5)
            for _ in range(total_blocks)
        ])

        self.readout_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        gate_input_dim = hidden_dim + num_classes + 6

        self.benefit_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

    def pool_feature(self, z):
        z = self.readout_norm(z)
        return z.mean(dim=1)

    def classify_from_feature(self, h):
        return self.classifier(h)

    def build_gate_input(self, x, h_early, early_logits):
        prob = F.softmax(early_logits, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values

        confidence = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.num_classes)

        temporal_energy = (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean(dim=(1, 2)).unsqueeze(1)
        signal_energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        abs_mean = x.abs().mean(dim=(1, 2)).unsqueeze(1)

        return torch.cat(
            [
                h_early,
                early_logits,
                confidence,
                margin,
                entropy,
                temporal_energy,
                signal_energy,
                abs_mean,
            ],
            dim=1,
        )

    @torch.no_grad()
    def forward_early_only(self, x):
        z = self.input_proj(x)
        z = z.transpose(1, 2)

        for i in range(self.early_exit_blocks):
            z = self.blocks[i](z)

        h = self.pool_feature(z)
        logits = self.classify_from_feature(h)
        return logits

    @torch.no_grad()
    def forward_full_only(self, x):
        z = self.input_proj(x)
        z = z.transpose(1, 2)

        for block in self.blocks:
            z = block(z)

        h = self.pool_feature(z)
        logits = self.classify_from_feature(h)
        return logits

    @torch.no_grad()
    def forward_dynamic(self, x, benefit_tau: float):
        z = self.input_proj(x)
        z = z.transpose(1, 2)

        for i in range(self.early_exit_blocks):
            z = self.blocks[i](z)

        h_early = self.pool_feature(z)
        early_logits = self.classify_from_feature(h_early)

        gate_input = self.build_gate_input(x, h_early, early_logits)
        gate_logit = self.benefit_gate(gate_input).squeeze(1)
        gate_prob = torch.sigmoid(gate_logit)

        full_mask = gate_prob >= benefit_tau

        output_logits = early_logits.clone()
        route = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if full_mask.any():
            z_full = z[full_mask]

            for i in range(self.early_exit_blocks, self.total_blocks):
                z_full = self.blocks[i](z_full)

            h_full = self.pool_feature(z_full)
            full_logits = self.classify_from_feature(h_full)

            output_logits[full_mask] = full_logits
            route[full_mask] = 1

        return output_logits, route, gate_prob


def benchmark_mode(model, X, mode, tau=None):
    indices = np.random.default_rng(SEED).choice(
        len(X),
        size=min(NUM_RUN_SAMPLES, len(X)),
        replace=False,
    )

    times = []
    routes = []

    # warm-up
    for idx in indices[:10]:
        x = torch.tensor(X[idx:idx + 1], dtype=torch.float32, device=DEVICE)
        if mode == "early":
            _ = model.forward_early_only(x)
        elif mode == "full":
            _ = model.forward_full_only(x)
        elif mode == "dynamic":
            _, route, _ = model.forward_dynamic(x, benefit_tau=tau)
        else:
            raise ValueError(mode)

    # measurement
    for idx in indices:
        x = torch.tensor(X[idx:idx + 1], dtype=torch.float32, device=DEVICE)

        for _ in range(REPEAT_PER_SAMPLE):
            start = time.perf_counter()

            if mode == "early":
                _ = model.forward_early_only(x)
            elif mode == "full":
                _ = model.forward_full_only(x)
            elif mode == "dynamic":
                _, route, _ = model.forward_dynamic(x, benefit_tau=tau)
                routes.append(int(route.item()))

            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    avg_latency = float(np.mean(times))

    if mode == "early":
        full_ratio = 0.0
    elif mode == "full":
        full_ratio = 1.0
    else:
        full_ratio = float(np.mean(routes))

    return avg_latency, full_ratio


def main():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    config = ckpt["config"]

    model = BenefitGatedSharedMambaHAR(
        in_channels=config["num_channels"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        total_blocks=config["total_mamba_blocks"],
        early_exit_blocks=config["early_exit_blocks"],
        dropout=config["dropout"],
        gate_hidden_dim=config["gate_hidden_dim"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X = np.load(X_PATH).astype(np.float32)

    tau = float(config["benefit_tau"])
    flops_info = config["flops_info"]

    print("Loaded model:", MODEL_PATH)
    print("Loaded X:", X.shape)
    print("Benefit tau:", tau)
    print("Threads:", torch.get_num_threads())
    print()

    early_latency, early_ratio = benchmark_mode(model, X, mode="early")
    full_latency, full_ratio = benchmark_mode(model, X, mode="full")
    dyn_latency, dyn_ratio = benchmark_mode(model, X, mode="dynamic", tau=tau)

    early_flops = flops_info["early_route"] / 1e6
    full_flops = flops_info["full_model_once"] / 1e6
    dyn_flops = (
        (1.0 - dyn_ratio) * flops_info["early_route"]
        + dyn_ratio * flops_info["full_route_dynamic"]
    ) / 1e6

    print("================ Raspberry Pi Inference Benchmark ================")
    print(f"{'Mode':<15} {'Avg.Latency(ms)':>18} {'FLOPs(M)':>12} {'Full Ratio':>12}")
    print("-" * 62)
    print(f"{'Early-only':<15} {early_latency:>18.4f} {early_flops:>12.4f} {early_ratio:>12.4f}")
    print(f"{'Full-only':<15} {full_latency:>18.4f} {full_flops:>12.4f} {full_ratio:>12.4f}")
    print(f"{'Proposed':<15} {dyn_latency:>18.4f} {dyn_flops:>12.4f} {dyn_ratio:>12.4f}")


if __name__ == "__main__":
    main()
