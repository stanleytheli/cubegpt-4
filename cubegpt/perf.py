"""Where a training step's time goes, and what speeds it up (modal_app.py::perf).

Everything is timed in one container so the numbers are comparable (hosts differ a lot).
A "step" is fwd + bwd + Adam + loss.item() + MoE bias update on pre-generated batches."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import torch
from torch import nn

from .config import ModelConfig
from .cube import CubeTables, generate_samples
from .model import build_model, count_params

LAYOUTS = {
    "moe8": dict(),
    "attn8-ffn4": dict(n_layers=12, ffn_layers="00000000ffff"),
    "attn4-moe4": dict(ffn_layers="00001111"),
    "attn7-moe1": dict(ffn_layers="00000001"),
    "dense8": dict(arch="gated"),
    "attnonly": dict(arch="gated", ffn_layers="00000000"),
    "attn8-dense4": dict(arch="gated", n_layers=12, ffn_layers="00000000ffff"),
    "attn8-dense6": dict(arch="gated", n_layers=14, ffn_layers="00000000ffffff"),
    "attn10-dense4": dict(arch="gated", n_layers=14, ffn_layers="0000000000ffff"),
    "a8f4a1f1": dict(arch="gated", n_layers=14, ffn_layers="00000000ffff0f"),
    "attn8-dense4-d384": dict(arch="gated", d_model=384, d_ffwd=512, n_layers=12, ffn_layers="00000000ffff"),
}


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def host_desc():
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count()
    return f"{cpu_name()}, {usable} usable cores"


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


class Stepper:
    def __init__(self, T, layout, batch_size=1024, fused=None, compile_mode=None, n_batches=64, parts="all"):
        torch.manual_seed(0)
        self.dev = T.device
        self.model = build_model(ModelConfig(**LAYOUTS[layout])).to(T.device).train()
        self.params = count_params(self.model)
        self.fwd = self.model
        if compile_mode is not None:
            self.fwd = torch.compile(self.model, **({} if compile_mode == "default" else {"mode": compile_mode}))
        kw = {} if fused is None else {"fused": fused}
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, **kw)
        self.loss_fn = nn.BCEWithLogitsLoss()
        X, y = generate_samples(T, n_batches * batch_size)
        self.X, self.y = X.view(n_batches, batch_size, 20, 2), y.view(n_batches, batch_size, -1)
        self.is_moe = hasattr(self.model, "update_expert_bias")
        self.parts = parts
        self.i = 0

    def step(self):
        j = self.i % len(self.X)
        self.i += 1
        loss = self.loss_fn(self.fwd(self.X[j]), self.y[j])
        if self.parts == "fwd":
            return
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.parts == "fwd+bwd":
            return
        self.opt.step()
        if self.parts == "fwd+bwd+opt":
            return
        loss.item()
        if self.is_moe:
            self.model.update_expert_bias()

    def time(self, seconds=5.0, warmup=20):
        for _ in range(warmup):
            self.step()
        sync(self.dev)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            self.step()
            n += 1
        sync(self.dev)
        return 1000 * (time.perf_counter() - t0) / n


def worker(layout, seconds, fused):
    T = CubeTables("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    s = Stepper(T, layout, fused=fused)
    s.time(1.0)
    print(f"READY", flush=True)
    sys.stdin.readline()                      # start together
    print(f"MS {s.time(seconds)}", flush=True)


def run_perf(spec, log=print):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    T = CubeTables(dev)
    secs = spec.get("seconds", 5.0)
    bs = spec.get("batch_size", 1024)
    out = {"host": host_desc(), "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "torch": torch.__version__, "batch_size": bs, "rows": []}
    log(f"{out['gpu']} | {out['host']} | torch {torch.__version__} | batch {bs}")

    def row(section, name, ms, **extra):
        r = {"section": section, "name": name, "ms_per_step": ms, "samples_per_sec": bs * 1000 / ms, **extra}
        out["rows"].append(r)
        log(f"[{section}] {name:<34} {ms:7.2f} ms/step {r['samples_per_sec']:>9,.0f} samples/s "
            + " ".join(f"{k}={v}" for k, v in extra.items()))

    # 1. layouts x optimizer implementation
    for layout in spec.get("layouts", list(LAYOUTS)):
        for fused in spec.get("fused", (None, True)):
            s = Stepper(T, layout, bs, fused=fused)
            row("layout", f"{layout} adam={'fused' if fused else 'default'}", s.time(secs),
                params=s.params, tensors=len(list(s.model.parameters())))
            del s

    # 2. cumulative breakdown of one step
    for layout in spec.get("breakdown", ["moe8", "attnonly"]):
        for parts in ("fwd", "fwd+bwd", "fwd+bwd+opt", "all"):
            for fused in (None, True):
                if parts in ("fwd", "fwd+bwd") and fused:
                    continue
                s = Stepper(T, layout, bs, fused=fused, parts=parts)
                row("breakdown", f"{layout} {parts} adam={'fused' if fused else 'default'}", s.time(secs))
                del s

    # 3. data generation, amortized per step
    for canonical in (True, False):
        n = 256 * bs
        generate_samples(T, n, canonical=canonical); sync(dev)
        t0 = time.perf_counter()
        generate_samples(T, n, canonical=canonical); sync(dev)
        row("datagen", f"canonical={canonical}", 1000 * (time.perf_counter() - t0) / 256)

    # 4. operator-level profile of the baseline step
    if spec.get("profile", True):
        from torch.profiler import ProfilerActivity, profile
        for layout, fused in (("moe8", None), ("moe8", True)):
            s = Stepper(T, layout, bs, fused=fused)
            s.time(1.0)
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
            with profile(activities=acts) as prof:
                for _ in range(20):
                    s.step()
                sync(dev)
            ka = prof.key_averages()
            calls = sum(e.count for e in ka)
            table = ka.table(sort_by="self_cpu_time_total", row_limit=22, max_name_column_width=44)
            log(f"--- profile {layout} adam={'fused' if fused else 'default'}: {calls / 20:.0f} op calls/step\n{table}")
            out[f"profile_{layout}_{'fused' if fused else 'default'}"] = table
            del s

    # 5. torch.compile
    for layout, mode in spec.get("compile", [("attnonly", "default"), ("attnonly", "reduce-overhead"),
                                             ("dense8", "default"), ("attn8-ffn4", "default"), ("moe8", "default")]):
        try:
            t0 = time.perf_counter()
            s = Stepper(T, layout, bs, fused=True, compile_mode=mode)
            s.time(0.5, warmup=30)
            row("compile", f"{layout} compile={mode} adam=fused", s.time(secs),
                compile_s=round(time.perf_counter() - t0 - 0.5, 1))
            del s
        except Exception as e:      # noqa: BLE001  (a benchmark: report and move on)
            log(f"[compile] {layout} compile={mode}: FAILED {type(e).__name__}: {str(e)[:300]}")
            out["rows"].append({"section": "compile", "name": f"{layout} compile={mode}", "error": str(e)[:300]})
        try:
            torch._dynamo.reset()
        except Exception:           # noqa: BLE001
            pass

    # 6. several runs sharing the GPU
    if dev == "cuda":
        for layout in spec.get("concurrent_layouts", ["moe8", "attn8-ffn4"]) if spec.get("concurrency", True) else []:
            for k in spec.get("concurrency", [1, 2, 4]):
                procs = [subprocess.Popen([sys.executable, "-m", "cubegpt.perf", layout, str(secs), "1"],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                         for _ in range(k)]
                try:
                    for p in procs:
                        assert p.stdout.readline().strip() == "READY"
                    for p in procs:
                        p.stdin.write("go\n"); p.stdin.flush()
                    ms = [float(p.stdout.readline().split()[1]) for p in procs]
                    total = sum(bs * 1000 / m for m in ms)
                    row("concurrent", f"{layout} x{k} processes (fused)", bs * 1000 / total * 1.0,
                        per_process_ms=[round(m, 2) for m in ms])
                except Exception as e:  # noqa: BLE001
                    log(f"[concurrent] {layout} x{k}: FAILED {type(e).__name__}: {e}")
                finally:
                    for p in procs:
                        p.kill()
    return out


if __name__ == "__main__":
    worker(sys.argv[1], float(sys.argv[2]), bool(int(sys.argv[3])))
