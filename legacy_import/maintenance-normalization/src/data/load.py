"""Loaders for the logbook, Amin's expert resources, and the built gold."""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_logbook(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["logbook"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    lc = cfg["logbook"]
    for col in (lc["id_col"], lc["problem_col"], lc["action_col"]):
        df[col] = df[col].fillna("").astype(str)
    return df


def combined_text(problem: str, action: str) -> str:
    """Combine PROBLEM + ACTION into one instance, matching Amin's cleaned unit."""
    p = (problem or "").strip()
    a = (action or "").strip()
    if p and a:
        sep = " " if p.endswith((".", "!", "?")) else ". "
        return p + sep + a
    return p or a


def load_expert_cleaned(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["expert_cleaned"], dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns][:5]
    df = df.rename(columns=dict(zip(df.columns, ["ID", "PROBLEM", "ACTION", "CLEANED", "EDIT"])))
    df = df[df["ID"].str.match(r"^\d+$", na=False)].reset_index(drop=True)
    return df


def load_abbreviations(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["expert_abbreviations"], dtype=str, keep_default_na=False)
    df.columns = ["abbrev", "expansion"]
    df["abbrev"] = df["abbrev"].str.strip()
    df["expansion"] = df["expansion"].str.strip().str.lower()
    return df


def load_misspellings(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["expert_misspellings"], dtype=str, keep_default_na=False)
    df.columns = ["misspelling", "correction"]
    df["misspelling"] = df["misspelling"].str.strip()
    # corrections sometimes carry a parenthetical note: "L/H (LEFT-HAND)" -> "L/H"
    df["correction"] = df["correction"].str.replace(r"\s*\(.*\)\s*", "", regex=True).str.strip()
    return df


def load_unexpanded(cfg: dict) -> list:
    df = pd.read_csv(cfg["paths"]["expert_unexpanded"], dtype=str, keep_default_na=False)
    return [x.strip().lower() for x in df.iloc[:, 0] if x.strip()]


def load_domain_vocab(cfg: dict) -> set:
    """Vocabulary of 'known' words for the OOV metric: every word that appears in
    Amin's cleaned (expert) text. Robust and corpus-grounded."""
    from src.utils.text import simple_tokens
    cl = load_expert_cleaned(cfg)
    vocab = set()
    for t in cl["CLEANED"]:
        vocab.update(simple_tokens(t))
    return vocab

