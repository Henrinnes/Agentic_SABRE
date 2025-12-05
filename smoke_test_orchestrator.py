# smoke_test_orchestrator.py
import json
from pathlib import Path

import numpy as np

from Agentic_SABRE.orchestrator import Orchestrator

if __name__ == "__main__":
    # 1) read input dims from metadata
    meta = json.loads(Path("checkpoints/metadata.json").read_text())
    d_sem = meta["semantic_input_dim"]
    d_beh = meta["behavioral_input_dim"]

    # 2) build one fake sample (correct dimensions)
    x_sem = {"embedding": np.random.rand(d_sem).astype("float32")}
    x_beh = {"features":  np.random.rand(d_beh).astype("float32")}

    # 3) create orchestrator using the trained decision_fuser.json
    orch = Orchestrator(
        τ=0.8,
        κ=0.15,
        τ_high=0.95,
        κ_low=0.08,
        decision_artifact="checkpoints/decision_fuser.json",
    )

    # 4) run the full pipeline
    out = orch.handle(
        x_sem,
        x_beh,
        context={"host": "lab-host", "pid": 1234},
    )

    print(out)
