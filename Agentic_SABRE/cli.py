import json, random
from pathlib import Path
import typer
from rich import print
from .orchestrator import Orchestrator

app = typer.Typer(add_completion=False)

@app.command()
def run(
    x_sem: Path,
    x_beh: Path,
    tau: float = 0.80,
    kappa: float = 0.15,
    tau_high: float = 0.95,
    kappa_low: float = 0.08,
):
    o = Orchestrator(τ=tau, κ=kappa, τ_high=tau_high, κ_low=kappa_low)
    xs = json.loads(x_sem.read_text())
    xb = json.loads(x_beh.read_text())
    print(o.handle(xs, xb, context={"host": "demo-host"}))

@app.command()
def demo(
    tau: float = 0.80,
    kappa: float = 0.15,
    tau_high: float = 0.95,
    kappa_low: float = 0.08,
):
    o = Orchestrator(τ=tau, κ=kappa, τ_high=tau_high, κ_low=kappa_low)
    xs = {"embedding": [random.random() for _ in range(128)]}   # dummy dims
    xb = {"features":  [random.random() for _ in range(16)]}
    print(o.handle(xs, xb, context={"host": "demo-host"}))
