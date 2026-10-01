"""
Knowledge Agent input definition: the single list of RansKnow columns the
Knowledge CNN consumes, and the columns that must never reach it.

The training label, Ransomware_Context, is defined from two RansKnow counts:

    Ransomware_Context = (Tactic_Total_Mentions > 10) | (Tool_Total_Mentions > 2)

Those two counts were previously also model inputs, so the CNN could recover
the label from its own features by a fixed rule rather than learn it. They are
now used only to build the label and are excluded from the inputs, leaving 21
features: the per-tactic, per-tool and per-platform counts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# RansKnow v5 (1,034 videos, 92 channels). SABRE_RANSKNOW_CSV overrides the file;
# otherwise it is looked up under SABRE_DATA_DIR like the other datasets.
RANSKNOW_CSV = os.environ.get(
    "SABRE_RANSKNOW_CSV",
    str(Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
        / "RansKnow" / "Knowledge_Agent_Features_1034.csv"),
)

LABEL_SOURCE_COLUMNS = ["Tactic_Total_Mentions", "Tool_Total_Mentions"]

KNOWLEDGE_FEATURES = [
    "Tactic_Initial_Access", "Tactic_Execution", "Tactic_Persistence",
    "Tactic_Privilege_Escalation", "Tactic_Credential_Access",
    "Tactic_Lateral_Movement", "Tactic_Discovery",
    "Tactic_Command_and_Control", "Tactic_Exfiltration", "Tactic_Impact",
    "Tool_Cobalt_Strike", "Tool_Mimikatz", "Tool_PsExec", "Tool_Rclone",
    "Tool_MegaNZ", "Tool_AnyDesk", "Tool_TeamViewer", "Tool_BloodHound",
    "Platform_Windows", "Platform_Linux", "Platform_ESXi",
]

# Every RansKnow column (v1-v5) that is not a model input. Scripts that build
# inputs as `df.drop(columns=...).select_dtypes("number")` use this list, which
# leaves exactly KNOWLEDGE_FEATURES, in file order.
KNOWLEDGE_NON_FEATURE_COLUMNS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Channel_Type", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Year", "DurationSeconds",
    "Transcript_Provider", "Family_Count", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal", "Ransomware_Context", "label",
] + LABEL_SOURCE_COLUMNS

assert not set(LABEL_SOURCE_COLUMNS) & set(KNOWLEDGE_FEATURES)


def ransomware_context_label(df: pd.DataFrame) -> pd.Series:
    """Binary Ransomware_Context label, from the label-source counts."""
    return ((df["Tactic_Total_Mentions"] > 10) | (df["Tool_Total_Mentions"] > 2)).astype(int)


def knowledge_matrix(df: pd.DataFrame) -> np.ndarray:
    """The Knowledge Agent's input matrix: KNOWLEDGE_FEATURES, in order."""
    return df[KNOWLEDGE_FEATURES].fillna(0).to_numpy(dtype=np.float32)
