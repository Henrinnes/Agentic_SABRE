import json, random
from pathlib import Path
import typer
from rich import print
from .orchestrator import Orchestrator


app = typer.Typer(add_completion=False)


@app.command()
def run(x_sem: Path, x_beh: Path, tau: float = 0.8, kappa: float = 0.15):
o = Orchestrator(τ=tau, κ=kappa)
xs = json.loads(Path(x_sem).read_text())
xb = json.loads(Path(x_beh).read_text())
print(o.handle(xs, xb, context={"host": "demo-host"}))


@app.command()
def demo(tau: float = 0.8, kappa: float = 0.15):
o = Orchestrator(τ=tau, κ=kappa)
xs = {"imports": ["CryptEncrypt", "WriteFile"], "size": 123456}
xb = {"entropy": random.uniform(0,1), "rename_rate": random.uniform(0,1), "io_burst": random.uniform(0,1)}
print(o.handle(xs, xb, context={"host": "demo-host"}))


if __name__ == "__main__":
app()
