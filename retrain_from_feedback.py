#!/usr/bin/env python
"""
retrain_from_feedback.py
========================
CLI: retrain the 3-agent decision MLP from analyst feedback.

Usage
-----
    python retrain_from_feedback.py

Options
-------
    --feedback_log      Path to feedback_log.json  [checkpoints/feedback/feedback_log.json]
    --base_artifact     Base decision artifact      [checkpoints/decision_fuser_3agent.json]
    --out               Output path for updated MLP [checkpoints/decision_fuser_3agent_updated.json]
    --feedback_repeat   Upweighting multiplier      [3]
    --min_samples       Minimum feedback records    [20]
    --retune_policy     Run Optuna threshold re-tune after retraining (requires datasets)
    --promote           Overwrite base artifact with updated one
    --dry_run           Show what would happen without writing files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Retrain decision MLP from expert feedback")
    parser.add_argument("--feedback_log",
                        default="checkpoints/feedback/feedback_log.json")
    parser.add_argument("--base_artifact",
                        default="checkpoints/decision_fuser_3agent.json")
    parser.add_argument("--out",
                        default="checkpoints/decision_fuser_3agent_updated.json")
    parser.add_argument("--feedback_repeat", type=int, default=3,
                        help="How many times feedback rows are duplicated for up-weighting")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="Minimum number of feedback records required before retraining")
    parser.add_argument("--retune_policy", action="store_true",
                        help="Re-tune global policy thresholds after retraining (requires Optuna)")
    parser.add_argument("--promote", action="store_true",
                        help="Replace base_artifact with the updated model after retraining")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load feedback
    # -----------------------------------------------------------------------
    from Agentic_SABRE.feedback_store import feedback_as_training_rows, feedback_count

    n = feedback_count(args.feedback_log)
    print(f"Feedback records found: {n}")

    if n < args.min_samples:
        print(
            f"ERROR: need at least {args.min_samples} records to retrain "
            f"(have {n}).  Collect more analyst corrections first."
        )
        sys.exit(1)

    X, y = feedback_as_training_rows(args.feedback_log)
    print(f"Training rows: X={X.shape}  y={y.shape}  "
          f"(pos={int(y.sum())}  neg={int((1-y).sum())})")

    if args.dry_run:
        print("[dry_run] Would retrain and save to:", args.out)
        if args.promote:
            print("[dry_run] Would promote to:", args.base_artifact)
        return

    # -----------------------------------------------------------------------
    # Retrain
    # -----------------------------------------------------------------------
    if not Path(args.base_artifact).exists():
        print(f"ERROR: base artifact not found: {args.base_artifact}")
        sys.exit(1)

    print(f"Retraining from base artifact: {args.base_artifact}")
    print(f"Feedback repeat multiplier: {args.feedback_repeat}×")

    from Agentic_SABRE.agents.decision import DecisionTrainer

    trainer = DecisionTrainer.retrain_incremental(
        base_artifact=args.base_artifact,
        X_feedback=X,
        y_feedback=y,
        out_path=args.out,
        feedback_repeat=args.feedback_repeat,
    )
    print(f"Updated model saved → {args.out}")

    # -----------------------------------------------------------------------
    # Optional: re-tune global policy thresholds
    # -----------------------------------------------------------------------
    if args.retune_policy:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("WARNING: optuna not installed — skipping policy re-tune.")
            args.retune_policy = False

    if args.retune_policy:
        print("Re-tuning global policy thresholds…")
        from eval_fusion_policy_3agent import run_policy_search  # noqa: F401
        print("  (Use tune_policy_optuna.py for full threshold re-tune)")

    # -----------------------------------------------------------------------
    # Optional: promote
    # -----------------------------------------------------------------------
    if args.promote:
        import shutil
        shutil.copy2(args.out, args.base_artifact)
        print(f"Promoted → {args.base_artifact}")

    print("Done.")


if __name__ == "__main__":
    main()
