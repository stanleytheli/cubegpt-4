"""Learning-rate plateau detection on the held-out eval loss.

Why not torch.optim.lr_scheduler.ReduceLROnPlateau stepped on per-step training loss
(the notebook's approach): it compares each single batch's loss with the lowest single
batch loss ever seen. At batch 256 per-batch noise is ~7% of the loss (hundreds of
times its 1e-4 threshold), so the record it chases is a lucky outlier, and "no
improvement for N steps" fires on noise statistics rather than real plateaus,
increasingly often the longer a stage runs, and at a rate that depends on batch size.

EvalPlateau instead watches the loss on the fixed held-out eval set (the same samples
every time, so it only moves when the model does; ~0.01% eval-to-eval jitter), fits a
least-squares line to the last `window_evals` evals, and decays the LR only when that
trend shows less than `threshold` relative improvement across the window. After each
decay it waits for a full fresh window at the new LR before judging again.
"""
from __future__ import annotations

import math


class EvalPlateau:
    def __init__(self, optimizer, factor, window_evals, threshold, min_lr=0.0):
        if not 0 < factor < 1:
            raise ValueError("factor must be in (0, 1)")
        if window_evals < 3:
            raise ValueError("window_evals must be >= 3")
        self.optimizer = optimizer
        self.factor = factor
        self.window_evals = window_evals
        self.threshold = threshold
        self.min_lr = min_lr
        self.hist = []            # (samples_seen, eval_loss) since the last LR change
        self.n_reductions = 0
        self.last_trend = None    # relative improvement across the last full window

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    def trend(points):
        """Relative improvement (fraction, positive = loss going down) across the span
        of `points` according to a least-squares line through them."""
        n = len(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        return -slope * (xs[-1] - xs[0]) / my if my else 0.0

    def observe(self, samples_seen, eval_loss):
        """Record one eval. Returns a log message if the LR was decayed, else None."""
        self.hist.append((float(samples_seen), float(eval_loss)))
        if len(self.hist) < self.window_evals:
            self.last_trend = None
            return None
        self.hist = self.hist[-self.window_evals:]
        t = self.trend(self.hist)
        self.last_trend = t
        if t >= self.threshold:
            return None
        old = self.lr
        new = max(old * self.factor, self.min_lr)
        if new >= old:
            return None   # already at the floor
        for g in self.optimizer.param_groups:
            g["lr"] = new * g.get("lr_mult", 1.0)
        self.hist = []
        self.n_reductions += 1
        return (f"plateau: eval loss improved {100 * t:.3f}% over the last {self.window_evals} evals "
                f"(threshold {100 * self.threshold:.3f}%) -> lr {old:.3g} -> {new:.3g}")

    def state_dict(self):
        return {"hist": [list(p) for p in self.hist], "n_reductions": self.n_reductions,
                "last_trend": self.last_trend}

    def load_state_dict(self, d):
        self.hist = [tuple(p) for p in d["hist"]]
        self.n_reductions = d["n_reductions"]
        self.last_trend = d["last_trend"]


def lr_factor(schedule, step, total_steps, warmup_steps=0, decay_steps=0, final_fraction=0.0):
    """Multiplier on the stage's peak lr for optimizer step `step` (0-based) of a
    fixed-shape schedule. Pure function of the step, so resuming is exact.

      "wsd":    linear warmup over warmup_steps, constant at the peak, then linear
                decay to final_fraction over the last decay_steps.
      "cosine": linear warmup, then cosine from the peak to final_fraction over the
                remaining steps.
    Warmup reaches the peak on its last step; decays reach final_fraction on the
    stage's last step."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "wsd":
        decay_start = total_steps - decay_steps
        if decay_steps <= 0 or step < decay_start:
            return 1.0
        p = (step - decay_start + 1) / decay_steps
    elif schedule == "cosine":
        span = total_steps - warmup_steps
        p = (step - warmup_steps + 1) / span
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    else:
        raise ValueError(f"unknown fixed schedule {schedule!r}")
    return 1.0 - (1.0 - final_fraction) * min(p, 1.0)


class StepSchedule:
    """Applies lr_factor() to the optimizer before every step. Stateless: the peak
    lr is the optimizer's lr at construction (the stage lr). Param groups may carry an
    "lr_mult" key; group 0 must be the main group (lr_mult 1)."""

    def __init__(self, optimizer, schedule, total_steps, warmup_steps, decay_steps, final_fraction):
        self.optimizer = optimizer
        self.schedule = schedule
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.final_fraction = final_fraction
        self.peak_lr = optimizer.param_groups[0]["lr"]

    def set(self, step):
        lr = self.peak_lr * lr_factor(self.schedule, step, self.total_steps, self.warmup_steps,
                                      self.decay_steps, self.final_fraction)
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)   # e.g. a slower router group (router_lr_mult)
        return lr

    def describe(self):
        tail = (f"constant, then linear decay over the last {self.decay_steps} steps"
                if self.schedule == "wsd" else f"cosine over {self.total_steps - self.warmup_steps} steps")
        return (f"{self.schedule}: peak lr {self.peak_lr:.3g}, warmup {self.warmup_steps} steps, {tail} "
                f"to {self.final_fraction:g}x peak")

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass
