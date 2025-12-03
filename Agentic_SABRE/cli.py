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




# Agentic_SABRE/cli.py
import json
import random
from pathlib import Path
import typer
from rich import print
import numpy as np

from .orchestrator import Orchestrator

app = typer.Typer(add_completion=False)

@app.command()
def demo(tau: float = 0.8, kappa: float = 0.15, tau_high: float = 0.95, kappa_low: float = 0.08):
    """
    Demo with random inputs (just to exercise the pipeline).
    """
    from pathlib import Path
    meta = json.loads(Path("checkpoints/metadata.json").read_text())
    d_sem = meta["semantic_input_dim"]
    d_beh = meta["behavioral_input_dim"]

    x_sem = {"embedding": np.random.rand(d_sem).astype("float32")}
    x_beh = {"features":  np.random.rand(d_beh).astype("float32")}

    orch = Orchestrator(tau=tau, kappa=kappa, tau_high=tau_high, kappa_low=kappa_low)
    out = orch.handle(x_sem, x_beh, context={"host": "demo-host"})
    print(out)

@app.command()
def eval_semantic(semantic_parquet: Path):
    """
    Simple evaluation of semantic agent alone: prints accuracy thresholded at 0.5.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = pd.read_parquet(semantic_parquet)
    X = np.stack(df["embedding"].values).astype("float32")
    y = df["label"].astype("int32").values

    from .agents.semantic import SemanticAgent
    agent = SemanticAgent()
    probs = []
    for v in X:
        out = agent.score({"embedding": v})
        probs.append(out.prob)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype("int32")
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    print({"accuracy": acc, "auc": auc})

if __name__ == "__main__":
    app()
