"""Muon (momentum + orthogonalized update) for the transformer stack's weight matrices,
with Adam / AdamW for everything else (embeddings, biases, LayerNorms, the output head).

Muon step for a matrix W with gradient G:
    buf  = momentum * buf + (1 - momentum) * G
    U    = G + momentum * (buf - G)            (Nesterov)
    O    = orthogonalize(U)                    (Newton-Schulz: singular values pushed to ~1)
    W   -= lr * sqrt(max(1, rows / cols)) * O
Every step therefore changes W by a fixed spectral size, whatever the gradient's scale.
Matrices of equal shape are orthogonalized together as one batch."""
import torch


def orthogonalize(G, steps=5):
    """Newton-Schulz iteration on a batch of matrices (..., rows, cols). Returns something close
    to U V^T of each matrix's SVD (singular values in roughly [0.7, 1.2], which is all Muon needs)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


class MuonAdam(torch.optim.Optimizer):
    """One optimizer object (so schedules, checkpoints and logs treat it like any other) whose
    param groups tagged "muon": True get the Muon update and whose other groups are stepped by an
    inner Adam / AdamW. Group learning rates are read every step, so external schedules work."""

    def __init__(self, groups, lr, momentum=0.95, ns_steps=5, betas=(0.9, 0.999), eps=1e-8, fused=False):
        groups = [dict(g) for g in groups]
        for g in groups:
            g.setdefault("muon", False)
            g.setdefault("weight_decay", 0.0)
        super().__init__(groups, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        self._adam_idx = [i for i, g in enumerate(self.param_groups) if not g["muon"]]
        self.adam = None
        if self._adam_idx:
            inner = [{"params": self.param_groups[i]["params"], "lr": self.param_groups[i]["lr"],
                      "weight_decay": self.param_groups[i]["weight_decay"]} for i in self._adam_idx]
            self.adam = torch.optim.AdamW(inner, lr=lr, betas=betas, eps=eps, weight_decay=0.0, fused=fused)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if not g["muon"]:
                continue
            by_shape = {}
            for p in g["params"]:
                if p.grad is not None:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, ps in by_shape.items():
                grads = torch.stack([p.grad for p in ps])
                st = self.state[ps[0]]            # one stacked momentum buffer per shape, kept on its first tensor
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(grads)
                buf = st["buf"].lerp_(grads, 1 - g["momentum"])
                upd = orthogonalize(grads.lerp(buf, g["momentum"]), g["ns_steps"])
                step = g["lr"] * max(1.0, shape[0] / shape[1]) ** 0.5
                for p, u in zip(ps, upd):
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(u, alpha=-step)
        if self.adam is not None:
            for ig, i in zip(self.adam.param_groups, self._adam_idx):
                ig["lr"] = self.param_groups[i]["lr"]
            self.adam.step()

    def state_dict(self):
        sd = super().state_dict()
        sd["adam"] = self.adam.state_dict() if self.adam is not None else None
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        inner = sd.pop("adam", None)
        super().load_state_dict(sd)
        if self.adam is not None and inner is not None:
            self.adam.load_state_dict(inner)


def muon_params(model):
    """The weight matrices Muon handles: 2-D weights inside the transformer stack (attention
    q/k/v/out and FFN projections). Embeddings, the head, biases and norms stay on Adam."""
    return [p for n, p in model.named_parameters() if n.startswith("stack.") and p.ndim == 2]
