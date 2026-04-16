# Agentic_SABRE/consultation.py
"""
Pure-function consultation engine for SABRE.

When an agent's uncertainty exceeds its local kappa threshold (kappa_local),
its probability is adjusted using inverse-uncertainty-weighted peer scores
before fusion. Uncertainty values are NEVER modified.

All functions are side-effect-free and depend only on their arguments.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .contracts import AgentOutput

_EPSILON = 1e-8

# Agent ordering used throughout (must be consistent with orchestrator)
AGENT_NAMES: List[str] = ["semantic", "behavioral", "knowledge"]

# Maps each agent to its two peers
_PEERS: Dict[str, List[str]] = {
    "semantic":   ["behavioral", "knowledge"],
    "behavioral": ["semantic",   "knowledge"],
    "knowledge":  ["semantic",   "behavioral"],
}


# ---------------------------------------------------------------------------
# Core scalar consultation
# ---------------------------------------------------------------------------

def consult_single(
    agent_name: str,
    outputs: Dict[str, AgentOutput],
    kappa_local: float,
) -> AgentOutput:
    """
    Adjust one agent's probability using peer scores if σ > kappa_local.

    Parameters
    ----------
    agent_name : str
        Name of the agent being evaluated ("semantic" / "behavioral" / "knowledge").
    outputs : dict[str, AgentOutput]
        Mapping of agent name → current AgentOutput for the sample.
    kappa_local : float
        Per-agent uncertainty threshold; consultation fires when
        outputs[agent_name].uncert > kappa_local.

    Returns
    -------
    AgentOutput
        Possibly updated AgentOutput.  If consultation did not fire, the
        original object is returned unchanged.  If it fired, a *new*
        AgentOutput is returned with:
          - prob  = inverse-uncertainty-weighted mean of self + peers
          - uncert = UNCHANGED (original agent uncertainty)
          - consulted = True
          - consultation_peers = list of peer names used
          - extras gains: "raw_prob", "adjusted_prob", "consultation_weights"
    """
    o = outputs[agent_name]
    if o.uncert <= kappa_local:
        return o

    # Collect self + peers
    participants = [agent_name] + _PEERS[agent_name]
    probs = np.array([outputs[n].prob   for n in participants], dtype=np.float64)
    sigmas = np.array([outputs[n].uncert for n in participants], dtype=np.float64)

    weights = 1.0 / (sigmas + _EPSILON)
    adjusted_prob = float(np.dot(weights, probs) / weights.sum())
    weight_dict = {n: float(w) for n, w in zip(participants, weights / weights.sum())}

    new_extras = dict(o.extras)
    new_extras["raw_prob"]               = float(o.prob)
    new_extras["adjusted_prob"]          = adjusted_prob
    new_extras["consultation_weights"]   = weight_dict

    return AgentOutput(
        prob=adjusted_prob,
        uncert=o.uncert,          # uncertainty is NEVER changed
        extras=new_extras,
        consulted=True,
        consultation_peers=_PEERS[agent_name],
        sample_id=o.sample_id,
    )


def apply_consultation(
    outputs: Dict[str, AgentOutput],
    policies: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, AgentOutput], Dict[str, bool]]:
    """
    Apply consultation to all three agents for a single sample.

    Parameters
    ----------
    outputs : dict[str, AgentOutput]
        {"semantic": AgentOutput, "behavioral": AgentOutput, "knowledge": AgentOutput}
    policies : dict[str, dict]
        Per-agent policy dict, e.g.:
        {"semantic": {"kappa_local": 0.2}, "behavioral": {"kappa_local": 0.3}, ...}

    Returns
    -------
    updated_outputs : dict[str, AgentOutput]
    triggered : dict[str, bool]
        Which agents had consultation applied.
    """
    updated: Dict[str, AgentOutput] = {}
    triggered: Dict[str, bool] = {}

    for name in AGENT_NAMES:
        kappa_local = float(policies.get(name, {}).get("kappa_local", 1.0))
        new_out = consult_single(name, outputs, kappa_local)
        updated[name]  = new_out
        triggered[name] = new_out.consulted

    return updated, triggered


# ---------------------------------------------------------------------------
# Batch vectorised consultation
# ---------------------------------------------------------------------------

def apply_consultation_batch(
    outputs_batch: List[Dict[str, AgentOutput]],
    policies: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, AgentOutput]], List[Dict[str, bool]]]:
    """
    Vectorised batch consultation using numpy boolean masks.

    For each agent:
      1. Build arrays P[B], S[B] for probs and sigmas.
      2. Build arrays P_peers[B,2], S_peers[B,2] for peer probs/sigmas.
      3. Identify samples where S[i] > kappa_local.
      4. For triggered samples, compute weighted mean in one pass.
      5. Rebuild AgentOutput objects only for triggered samples.

    Parameters
    ----------
    outputs_batch : list of dicts
        Each element is {"semantic": AgentOutput, "behavioral": AgentOutput,
                         "knowledge": AgentOutput} for one sample.
    policies : dict[str, dict]
        Per-agent policy thresholds.

    Returns
    -------
    updated_batch : list of dicts (same length as outputs_batch)
    triggered_batch : list of dicts, each {"semantic": bool, ...}
    """
    B = len(outputs_batch)
    if B == 0:
        return [], []

    # Pre-extract all probs and sigmas into arrays [3, B]
    prob_mat  = np.array(
        [[outputs_batch[i][n].prob   for i in range(B)] for n in AGENT_NAMES],
        dtype=np.float64,
    )  # shape [3, B]
    sigma_mat = np.array(
        [[outputs_batch[i][n].uncert for i in range(B)] for n in AGENT_NAMES],
        dtype=np.float64,
    )  # shape [3, B]

    # Result containers — start as copies
    updated_batch   = [{n: outputs_batch[i][n] for n in AGENT_NAMES} for i in range(B)]
    triggered_batch = [{n: False               for n in AGENT_NAMES} for i in range(B)]

    for a_idx, agent_name in enumerate(AGENT_NAMES):
        kappa_local = float(policies.get(agent_name, {}).get("kappa_local", 1.0))
        peer_names  = _PEERS[agent_name]
        peer_idxs   = [AGENT_NAMES.index(p) for p in peer_names]

        # Rows: self + 2 peers → shape [3, B]
        all_idxs  = [a_idx] + peer_idxs
        p_stack   = prob_mat[all_idxs, :]    # [3, B]
        s_stack   = sigma_mat[all_idxs, :]   # [3, B]
        w_stack   = 1.0 / (s_stack + _EPSILON)  # [3, B]

        # Weighted mean: [B]
        adjusted  = (w_stack * p_stack).sum(axis=0) / w_stack.sum(axis=0)

        # Boolean mask of triggered samples
        mask = sigma_mat[a_idx, :] > kappa_local  # [B]

        if not mask.any():
            continue

        # Rebuild only triggered outputs
        triggered_idxs = np.where(mask)[0]
        for i in triggered_idxs:
            o = outputs_batch[i][agent_name]
            wn = w_stack[:, i] / w_stack[:, i].sum()
            weight_dict = {n: float(w) for n, w in zip([agent_name] + peer_names, wn)}

            new_extras = dict(o.extras)
            new_extras["raw_prob"]             = float(o.prob)
            new_extras["adjusted_prob"]        = float(adjusted[i])
            new_extras["consultation_weights"] = weight_dict

            updated_batch[i][agent_name] = AgentOutput(
                prob=float(adjusted[i]),
                uncert=o.uncert,
                extras=new_extras,
                consulted=True,
                consultation_peers=peer_names,
                sample_id=o.sample_id,
            )
            triggered_batch[i][agent_name] = True

    return updated_batch, triggered_batch
