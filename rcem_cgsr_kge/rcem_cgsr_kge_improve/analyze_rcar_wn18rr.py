#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis script for RCAR Sections 4.4 and 4.5.

This script uses an already trained WN18RR checkpoint to produce the empirical
materials needed for:

4.4 Analysis of Relation-Context Routing
  - average routing weights across relation types
  - case-level routing behavior

4.5 Analysis of Relation-Conditioned Evidence Calibration
  - MRR of full / w-o path / w-o role / w-o both evidence by relation type
  - average evidence gates across relation types
  - optional qualitative cases where evidence calibration improves ranks

Expected project layout:
  project/
    datasets.py
    models.py
    data/WN18RR/{train.pickle,valid.pickle,test.pickle,to_skip.pickle,rel_id,ent_id}
    logs/<run_dir>/{checkpoint,config.json}

Example:
  python analyze_rcar_wn18rr.py \
      --data_path ../../data \
      --dataset WN18RR \
      --checkpoint ./logs/MSDCSE_NA_WN18RR_rcar \
      --out_dir ./analysis_wn18rr \
      --batch_size 500

If config.json is found in the checkpoint directory, model hyperparameters are
read from it. Command-line arguments override only analysis options unless
--override_config is used for model options.
"""

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from models import MSDCSE


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_filter_size_list(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_id_file(path: str) -> Dict[int, str]:
    """Read files like rel_id / ent_id: name<TAB>id."""
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            name, idx = parts
            try:
                mapping[int(idx)] = name
            except ValueError:
                continue
    return mapping


def inverse_relation_name(name: str) -> str:
    return name + "^{-1}"


def load_names(data_root: str, dataset: str, real_r: int) -> Tuple[Dict[int, str], Dict[int, str]]:
    root = os.path.join(data_root, dataset)
    rel_names = read_id_file(os.path.join(root, "rel_id"))
    ent_names = read_id_file(os.path.join(root, "ent_id"))
    # Add inverse relation names for reciprocal relations.
    for r in range(real_r):
        base = rel_names.get(r, f"rel_{r}")
        rel_names[r + real_r] = inverse_relation_name(base)
    return rel_names, ent_names


def label_from_rel_stats(rel_stats_row: np.ndarray) -> str:
    # build_relation_context stores one-hot mapping type in columns 5:9:
    # 0: one-to-one, 1: one-to-many, 2: many-to-one, 3: many-to-many.
    labels = ["1-to-1", "1-to-N", "N-to-1", "N-to-N"]
    if rel_stats_row.shape[0] < 9:
        return "unknown"
    idx = int(np.argmax(rel_stats_row[5:9]))
    return labels[idx]


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(checkpoint: str) -> Dict:
    ckpt_path = Path(checkpoint)
    ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def default_model_args() -> Dict:
    return {
        "rank": 300,
        "input_drop": 0.4,
        "hidden_drop": 0.3,
        "feature_map_drop": 0.3,
        "k_w": 10,
        "k_h": 20,
        "output_channel": 20,
        "filter_size_list": [(1, 5, 1, 2), (3, 3), (1, 9)],
        "active_fn": "relu",
        "init_fn": "xavier_normal",
        "use_scale_router": True,
        "module_warmup_epochs": 0,
        "module_ramp_epochs": 1,
        "router_hidden": 0,
        "router_dropout": 0.1,
        "router_temperature": 1.0,
        "router_min_branch_weight": 0.0,
        "use_router_residual": False,
        "router_residual_init": 0.10,
        "no_router_query_context": False,
        "no_router_energy_preserving": False,
        "relation_context_dim": 0,
        "relation_context_hidden": 0,
        "relation_context_dropout": 0.10,
        "use_rcem": True,
        "rcem_no_path": False,
        "rcem_no_type": False,
        "rcem_max_rules": 8,
        "rcem_max_candidates": 32,
        "rcem_min_rule_support": 3,
        "rcem_max_rule_degree": 64,
        "rcem_warmup_epochs": 0,
        "rcem_ramp_epochs": 5,
        "rcem_gate_hidden": 0,
        "rcem_gate_dropout": 0.05,
        "rcem_path_strength": 0.10,
        "rcem_type_strength": 0.04,
        "rcem_path_gate_init": 0.05,
        "rcem_type_gate_init": 0.05,
        "rcem_no_query_gate": False,
        "rcem_no_candidate_calibrator": False,
        "rcem_calibrator_hidden": 0,
        "rcem_calibrator_strength": 0.25,
        "rcem_allow_inverse_paths": False,
    }


def resolve_checkpoint_file(checkpoint: str) -> str:
    p = Path(checkpoint)
    if p.is_dir():
        p = p / "checkpoint"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return str(p)


def clean_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def build_model_from_checkpoint(args, dataset: Dataset, config: Dict, device: torch.device):
    # get_train() is required before get_shape(), because reciprocal relations
    # double the number of predicates in this code base.
    dataset.get_train()

    model_cfg = default_model_args()
    model_cfg.update({k: v for k, v in config.items() if k in model_cfg})

    # Optional model-level overrides. Use this only when no config.json exists
    # or when you intentionally want to reconstruct the model manually.
    if args.override_config:
        for key in model_cfg.keys():
            if hasattr(args, key) and getattr(args, key) is not None:
                model_cfg[key] = getattr(args, key)

    relation_context = dataset.build_relation_context() if (model_cfg["use_scale_router"] or model_cfg["use_rcem"]) else None
    rcem_context = None
    if model_cfg["use_rcem"]:
        rcem_context = dataset.build_rcem_context(
            max_rules_per_relation=int(model_cfg["rcem_max_rules"]),
            max_candidates_per_query=int(model_cfg["rcem_max_candidates"]),
            min_rule_support=int(model_cfg["rcem_min_rule_support"]),
            max_rule_degree=int(model_cfg["rcem_max_rule_degree"]),
            use_path=not bool(model_cfg["rcem_no_path"]),
            use_type=not bool(model_cfg["rcem_no_type"]),
            exclude_inverse_paths=not bool(model_cfg["rcem_allow_inverse_paths"]),
        )

    filter_size_list = parse_filter_size_list(model_cfg["filter_size_list"])
    router_hidden = int(model_cfg["router_hidden"]) if int(model_cfg["router_hidden"]) > 0 else None
    rcem_gate_hidden = int(model_cfg["rcem_gate_hidden"]) if int(model_cfg["rcem_gate_hidden"]) > 0 else None
    relation_context_dim = int(model_cfg["relation_context_dim"]) if int(model_cfg["relation_context_dim"]) > 0 else None
    relation_context_hidden = int(model_cfg["relation_context_hidden"]) if int(model_cfg["relation_context_hidden"]) > 0 else None
    rcem_calibrator_hidden = int(model_cfg["rcem_calibrator_hidden"]) if int(model_cfg["rcem_calibrator_hidden"]) > 0 else None

    model = MSDCSE(
        num_ent=dataset.get_shape()[0],
        num_rel=dataset.get_shape()[1],
        embedding_dim=int(model_cfg["rank"]),
        input_drop=float(model_cfg["input_drop"]),
        hidden_drop=float(model_cfg["hidden_drop"]),
        feature_map_drop=float(model_cfg["feature_map_drop"]),
        k_w=int(model_cfg["k_w"]),
        k_h=int(model_cfg["k_h"]),
        output_channel=int(model_cfg["output_channel"]),
        filter_size_list=filter_size_list,
        active_fn=model_cfg["active_fn"],
        init_fn=model_cfg["init_fn"],
        ce_weight=None,
        use_scale_router=bool(model_cfg["use_scale_router"]),
        relation_context=relation_context,
        module_warmup_epochs=int(model_cfg["module_warmup_epochs"]),
        module_ramp_epochs=int(model_cfg["module_ramp_epochs"]),
        router_hidden=router_hidden,
        router_dropout=float(model_cfg["router_dropout"]),
        router_temperature=float(model_cfg["router_temperature"]),
        router_min_branch_weight=float(model_cfg["router_min_branch_weight"]),
        router_residual=bool(model_cfg["use_router_residual"]),
        router_residual_init=float(model_cfg["router_residual_init"]),
        router_use_query_context=not bool(model_cfg["no_router_query_context"]),
        router_energy_preserving=not bool(model_cfg["no_router_energy_preserving"]),
        relation_context_dim=relation_context_dim,
        relation_context_hidden=relation_context_hidden,
        relation_context_dropout=float(model_cfg["relation_context_dropout"]),
        use_rcem=bool(model_cfg["use_rcem"]),
        rcem_context=rcem_context,
        rcem_use_path=not bool(model_cfg["rcem_no_path"]),
        rcem_use_type=not bool(model_cfg["rcem_no_type"]),
        rcem_warmup_epochs=int(model_cfg["rcem_warmup_epochs"]),
        rcem_ramp_epochs=int(model_cfg["rcem_ramp_epochs"]),
        rcem_gate_hidden=rcem_gate_hidden,
        rcem_gate_dropout=float(model_cfg["rcem_gate_dropout"]),
        rcem_path_strength=float(model_cfg["rcem_path_strength"]),
        rcem_type_strength=float(model_cfg["rcem_type_strength"]),
        rcem_path_gate_init=float(model_cfg["rcem_path_gate_init"]),
        rcem_type_gate_init=float(model_cfg["rcem_type_gate_init"]),
        rcem_use_query_gate=not bool(model_cfg["rcem_no_query_gate"]),
        rcem_use_candidate_calibrator=not bool(model_cfg["rcem_no_candidate_calibrator"]),
        rcem_calibrator_hidden=rcem_calibrator_hidden,
        rcem_calibrator_strength=float(model_cfg["rcem_calibrator_strength"]),
    )

    model.to(device)
    # The original MSDCSE class stores a device attribute used in to_var().
    if hasattr(model, "device"):
        model.device = device
    model.eval()

    ckpt_file = resolve_checkpoint_file(args.checkpoint)
    state = torch.load(ckpt_file, map_location=device)
    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=not args.non_strict)
    if args.non_strict:
        print("[WARN] non-strict loading")
        print("  missing keys:", missing)
        print("  unexpected keys:", unexpected)

    model.eval()
    return model, relation_context, rcem_context, model_cfg


# -----------------------------------------------------------------------------
# 4.4 Routing analysis
# -----------------------------------------------------------------------------


def get_relation_ids(dataset: Dataset, include_inverse: bool) -> np.ndarray:
    if include_inverse:
        return np.arange(dataset.n_predicates, dtype=np.int64)
    return np.arange(dataset.real_r, dtype=np.int64)


@torch.no_grad()
def extract_routing_weights(model, rel_ids: np.ndarray, device: torch.device, normalize_residual: bool = True) -> np.ndarray:
    if getattr(model, "scale_router", None) is None:
        raise RuntimeError("The loaded model has no scale_router. Train or load a checkpoint with --use_scale_router.")
    rel_tensor = torch.as_tensor(rel_ids, dtype=torch.long, device=device)
    rel_emb = model.emb_rel(rel_tensor)
    rel_stats = model.rel_stats[rel_tensor] if getattr(model, "rel_stats", torch.empty(0)).numel() > 0 else None
    context_emb = None
    if getattr(model, "relation_context_encoder", None) is not None:
        context_emb = model.relation_context_encoder(rel_emb, rel_stats)
    query_emb = torch.zeros_like(rel_emb)
    alpha = model.scale_router(
        rel_emb,
        rel_stats=rel_stats,
        query_emb=query_emb,
        context_emb=context_emb,
    ).detach().cpu().numpy()
    # If the router was trained in residual-gain mode, alpha may be branch gains
    # rather than probabilities. For plotting relative branch preference, normalize.
    if normalize_residual:
        row_sum = np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        alpha = alpha / row_sum
    return alpha


def summarize_by_group(values: np.ndarray, groups: List[str], col_prefix: str) -> Tuple[List[Dict], List[str]]:
    rows = []
    unique_groups = ["1-to-1", "1-to-N", "N-to-1", "N-to-N", "unknown"]
    n_col = values.shape[1]
    fieldnames = ["relation_group", "count"] + [f"{col_prefix}{i+1}" for i in range(n_col)]
    for g in unique_groups:
        idx = [i for i, x in enumerate(groups) if x == g]
        if not idx:
            continue
        mean = values[idx].mean(axis=0)
        row = {"relation_group": g, "count": len(idx)}
        for j in range(n_col):
            row[f"{col_prefix}{j+1}"] = float(mean[j])
        rows.append(row)
    return rows, fieldnames


def select_case_relations(rel_names: Dict[int, str], rel_ids: np.ndarray, alpha: np.ndarray, user_cases: str) -> List[int]:
    if user_cases:
        wanted = [x.strip() for x in user_cases.split(",") if x.strip()]
    else:
        wanted = [
            "similar_to", "also_see", "hypernym", "has_part",
            "member_meronym", "derivationally_related_form"
        ]

    selected = []
    lower_name = {r: rel_names.get(int(r), f"rel_{r}").lower() for r in rel_ids}
    for w in wanted:
        wl = w.lower()
        exact = [r for r in rel_ids if lower_name[int(r)] == wl]
        fuzzy = [r for r in rel_ids if wl in lower_name[int(r)]]
        candidates = exact or fuzzy
        for r in candidates:
            if int(r) not in selected:
                selected.append(int(r))
                break

    # Fallback: choose relations with the strongest branch preference.
    if len(selected) < 5:
        preference = np.max(alpha, axis=1) - np.min(alpha, axis=1)
        order = np.argsort(-preference)
        for idx in order:
            r = int(rel_ids[idx])
            if r not in selected:
                selected.append(r)
            if len(selected) >= 5:
                break
    return selected[:6]


# -----------------------------------------------------------------------------
# 4.5 Evidence analysis
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_evidence_gates(model, rel_ids: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    if getattr(model, "rcem", None) is None:
        raise RuntimeError("The loaded model has no RelationEvidenceMemory. Train or load a checkpoint with --use_rcem.")
    rel_tensor = torch.as_tensor(rel_ids, dtype=torch.long, device=device)
    rel_emb = model.emb_rel(rel_tensor)
    rel_stats = model.rel_stats[rel_tensor] if getattr(model, "rel_stats", torch.empty(0)).numel() > 0 else None
    context_emb = None
    if getattr(model, "relation_context_encoder", None) is not None:
        context_emb = model.relation_context_encoder(rel_emb, rel_stats)
    head_emb = torch.zeros_like(rel_emb)
    path_quality = torch.zeros(rel_emb.size(0), 5, device=device, dtype=rel_emb.dtype)
    if getattr(model.rcem, "type_quality", torch.empty(0)).numel() > 0:
        type_quality = model.rcem.type_quality[rel_tensor]
    else:
        type_quality = torch.zeros(rel_emb.size(0), 4, device=device, dtype=rel_emb.dtype)
    quality = torch.cat([path_quality, type_quality], dim=1)
    gate_input = model.rcem._gate_input(rel_emb, rel_stats, head_emb, context_emb, quality)
    gate_logits = model.rcem.gate(gate_input)
    gate_logits = gate_logits.masked_fill(
        ~model.rcem.gate_active_mask.view(1, 3), -1e4
    )
    gates = torch.softmax(gate_logits, dim=-1).detach().cpu().numpy()
    effective = np.stack([
        gates[:, 1] * float(model.rcem.path_strength),
        gates[:, 2] * float(model.rcem.type_strength),
    ], axis=1)
    return gates, effective


def set_rcem_variant(model, use_path: bool, use_type: bool):
    if getattr(model, "rcem", None) is None:
        return
    model.rcem.use_path = bool(use_path)
    model.rcem.use_type = bool(use_type)


@torch.no_grad()
def filtered_ranks_for_queries(model, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]], batch_size: int) -> torch.Tensor:
    ranks = torch.ones(len(queries), device="cpu")
    n = len(queries)
    for b in range(0, n, batch_size):
        q = queries[b:b + batch_size]
        target_ids = q[:, 2].detach().cpu().tolist()
        scores, _ = model.forward(q)
        targets = torch.stack([scores[i, t] for i, t in enumerate(target_ids)]).unsqueeze(1)

        for i, query in enumerate(q):
            h = int(query[0].item())
            r = int(query[1].item())
            t = int(query[2].item())
            filter_out = list(filters.get((h, r), []))
            filter_out.append(t)
            scores[i, torch.as_tensor(filter_out, dtype=torch.long, device=scores.device)] = -1e6

        ranks[b:b + batch_size] = (1.0 + torch.sum((scores >= targets).float(), dim=1)).detach().cpu()
    return ranks


def build_eval_queries(dataset: Dataset, split: str, device: torch.device, directions: str) -> List[Tuple[str, torch.Tensor, Dict]]:
    examples = torch.from_numpy(dataset.get_examples(split).astype("int64")).to(device)
    items = []
    if directions in ("rhs", "both"):
        q_rhs = examples.clone()
        items.append(("rhs", q_rhs, dataset.to_skip["rhs"]))
    if directions in ("lhs", "both"):
        q_lhs = examples.clone()
        tmp = q_lhs[:, 0].clone()
        q_lhs[:, 0] = q_lhs[:, 2]
        q_lhs[:, 2] = tmp
        q_lhs[:, 1] += dataset.real_r
        items.append(("lhs", q_lhs, dataset.to_skip["lhs"]))
    return items


def ranks_by_relation_group(dataset: Dataset, model, split: str, batch_size: int, device: torch.device,
                            rel_group: Dict[int, str], directions: str) -> Dict[str, np.ndarray]:
    out = defaultdict(list)
    for name, queries, filters in build_eval_queries(dataset, split, device, directions):
        ranks = filtered_ranks_for_queries(model, queries, filters, batch_size=batch_size).numpy()
        rels = queries[:, 1].detach().cpu().numpy()
        for rank, r in zip(ranks, rels):
            out[rel_group[int(r)]].append(float(rank))
            out["Overall"].append(float(rank))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def metric_from_ranks(ranks: np.ndarray) -> Dict[str, float]:
    if len(ranks) == 0:
        return {"MRR": np.nan, "Hits@1": np.nan, "Hits@3": np.nan, "Hits@10": np.nan, "count": 0}
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "count": int(len(ranks)),
    }


def evaluate_evidence_variants(dataset, model, split: str, batch_size: int, device: torch.device,
                               rel_group: Dict[int, str], directions: str) -> List[Dict]:
    if getattr(model, "rcem", None) is None:
        raise RuntimeError("Evidence variants require model.rcem.")

    original_path = bool(model.rcem.use_path)
    original_type = bool(model.rcem.use_type)

    variants = [
        ("Full RCAR", True, True),
        ("w/o path", False, True),
        ("w/o role", True, False),
        ("w/o both", False, False),
    ]
    all_results = {}
    for variant_name, use_path, use_type in variants:
        set_rcem_variant(model, use_path, use_type)
        by_group = ranks_by_relation_group(dataset, model, split, batch_size, device, rel_group, directions)
        all_results[variant_name] = {g: metric_from_ranks(ranks) for g, ranks in by_group.items()}

    set_rcem_variant(model, original_path, original_type)

    group_order = ["1-to-1", "1-to-N", "N-to-1", "N-to-N", "Overall"]
    rows = []
    for group in group_order:
        row = {"relation_group": group}
        for variant_name, _, _ in variants:
            m = all_results.get(variant_name, {}).get(group, None)
            row[variant_name] = np.nan if m is None else m["MRR"]
            row[variant_name + "_count"] = 0 if m is None else m["count"]
        rows.append(row)
    return rows


@torch.no_grad()
def collect_case_study(dataset: Dataset, model, split: str, batch_size: int, device: torch.device,
                       rel_group: Dict[int, str], rel_names: Dict[int, str], ent_names: Dict[int, str],
                       directions: str, max_cases: int = 20) -> List[Dict]:
    if getattr(model, "rcem", None) is None:
        return []

    original_path = bool(model.rcem.use_path)
    original_type = bool(model.rcem.use_type)

    rows = []
    for direction, queries, filters in build_eval_queries(dataset, split, device, directions):
        # Full ranks
        set_rcem_variant(model, True, True)
        full_ranks = filtered_ranks_for_queries(model, queries, filters, batch_size).numpy()
        # Base ranks without evidence
        set_rcem_variant(model, False, False)
        base_ranks = filtered_ranks_for_queries(model, queries, filters, batch_size).numpy()
        # Remove one evidence source at a time for attribution.
        set_rcem_variant(model, False, True)
        no_path_ranks = filtered_ranks_for_queries(model, queries, filters, batch_size).numpy()
        set_rcem_variant(model, True, False)
        no_role_ranks = filtered_ranks_for_queries(model, queries, filters, batch_size).numpy()

        q_np = queries.detach().cpu().numpy()
        improvements = base_ranks - full_ranks
        order = np.argsort(-improvements)
        for idx in order:
            if improvements[idx] <= 0:
                continue
            h, r, t = map(int, q_np[idx])
            # If removing path makes rank worse than removing role, path evidence is more important.
            path_contrib = no_path_ranks[idx] - full_ranks[idx]
            role_contrib = no_role_ranks[idx] - full_ranks[idx]
            if path_contrib > role_contrib + 1e-6:
                main_evidence = "path"
            elif role_contrib > path_contrib + 1e-6:
                main_evidence = "role"
            else:
                main_evidence = "path + role"
            rows.append({
                "direction": direction,
                "head_id": h,
                "head": ent_names.get(h, f"ent_{h}"),
                "relation_id": r,
                "relation": rel_names.get(r, f"rel_{r}"),
                "tail_id": t,
                "target": ent_names.get(t, f"ent_{t}"),
                "relation_group": rel_group.get(r, "unknown"),
                "base_rank": int(base_ranks[idx]),
                "full_rank": int(full_ranks[idx]),
                "rank_gain": int(base_ranks[idx] - full_ranks[idx]),
                "rank_without_path": int(no_path_ranks[idx]),
                "rank_without_role": int(no_role_ranks[idx]),
                "main_evidence": main_evidence,
            })
            if len(rows) >= max_cases:
                break
        if len(rows) >= max_cases:
            break

    set_rcem_variant(model, original_path, original_type)
    rows.sort(key=lambda x: (-x["rank_gain"], x["full_rank"]))
    return rows[:max_cases]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_heatmap(values: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, out_base: str):
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(labels: List[str], values: np.ndarray, series_names: List[str], title: str, ylabel: str, out_base: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    x = np.arange(len(labels))
    n_series = values.shape[1]
    width = 0.8 / max(n_series, 1)
    for j in range(n_series):
        ax.bar(x + (j - (n_series - 1) / 2.0) * width, values[:, j], width, label=series_names[j])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../data")
    parser.add_argument("--dataset", default="WN18RR")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint directory containing 'checkpoint', or the checkpoint file itself.")
    parser.add_argument("--out_dir", default="./analysis_wn18rr")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--directions", default="both", choices=["rhs", "lhs", "both"])
    parser.add_argument("--include_inverse_relations", action="store_true",
                        help="Include inverse relations in routing/gate summaries. Default uses original relations only for figures.")
    parser.add_argument("--normalize_router", action="store_true", default=True,
                        help="Normalize branch values row-wise. Useful if the router uses residual gains.")
    parser.add_argument("--case_relations", default="",
                        help="Comma-separated relation names for Fig. 5. Fuzzy matching is used.")
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--non_strict", action="store_true")
    parser.add_argument("--override_config", action="store_true",
                        help="Override model hyperparameters from CLI instead of config.json. Mostly for debugging.")

    # Manual model options used only when --override_config is set or config.json is absent.
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--input_drop", type=float, default=None)
    parser.add_argument("--hidden_drop", type=float, default=None)
    parser.add_argument("--feature_map_drop", type=float, default=None)
    parser.add_argument("--k_w", type=int, default=None)
    parser.add_argument("--k_h", type=int, default=None)
    parser.add_argument("--output_channel", type=int, default=None)
    parser.add_argument("--filter_size_list", default=None)
    parser.add_argument("--active_fn", default=None)
    parser.add_argument("--init_fn", default=None)
    parser.add_argument("--use_scale_router", action="store_true")
    parser.add_argument("--use_rcem", action="store_true")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This code base creates some tensors on CUDA inside MSDCSE. "
            "Run the analysis on a machine with CUDA, or first make the model code device-agnostic."
        )
    device = torch.device("cuda")

    config = load_config(args.checkpoint)
    if config:
        print(f"Loaded config.json from checkpoint directory with {len(config)} entries.")
    else:
        print("[WARN] No config.json found. Falling back to default model arguments. Use --override_config to pass options manually.")

    dataset = Dataset(args.data_path, args.dataset)
    model, relation_context, rcem_context, model_cfg = build_model_from_checkpoint(args, dataset, config, device)

    rel_names, ent_names = load_names(args.data_path, args.dataset, dataset.real_r)
    rel_stats_np = model.rel_stats.detach().cpu().numpy() if getattr(model, "rel_stats", torch.empty(0)).numel() > 0 else None
    if rel_stats_np is None or rel_stats_np.shape[1] < 9:
        raise RuntimeError("Relation statistics are missing. The analysis requires dataset.build_relation_context().")

    rel_group = {r: label_from_rel_stats(rel_stats_np[r]) for r in range(dataset.n_predicates)}
    rel_ids_for_fig = get_relation_ids(dataset, include_inverse=args.include_inverse_relations)
    groups_for_fig = [rel_group[int(r)] for r in rel_ids_for_fig]

    save_json(os.path.join(args.out_dir, "analysis_config.json"), {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "directions": args.directions,
        "include_inverse_relations": args.include_inverse_relations,
        "model_config_used": model_cfg,
    })

    # ------------------------- 4.4 Routing analysis --------------------------
    alpha = extract_routing_weights(model, rel_ids_for_fig, device, normalize_residual=args.normalize_router)
    branch_names = [f"Scale-{i+1}" for i in range(alpha.shape[1])]

    routing_rows = []
    for idx, r in enumerate(rel_ids_for_fig):
        row = {
            "rel_id": int(r),
            "relation": rel_names.get(int(r), f"rel_{int(r)}"),
            "relation_group": rel_group[int(r)],
        }
        for j in range(alpha.shape[1]):
            row[f"alpha_{j+1}"] = float(alpha[idx, j])
        routing_rows.append(row)
    write_csv(
        os.path.join(args.out_dir, "routing_by_relation.csv"),
        routing_rows,
        ["rel_id", "relation", "relation_group"] + [f"alpha_{j+1}" for j in range(alpha.shape[1])]
    )

    routing_group_rows, routing_group_fields = summarize_by_group(alpha, groups_for_fig, "alpha_")
    write_csv(os.path.join(args.out_dir, "routing_by_type.csv"), routing_group_rows, routing_group_fields)

    heat_rows = []
    heat_labels = []
    for row in routing_group_rows:
        heat_labels.append(row["relation_group"])
        heat_rows.append([row[f"alpha_{j+1}"] for j in range(alpha.shape[1])])
    if heat_rows:
        plot_heatmap(
            np.asarray(heat_rows, dtype=np.float32),
            heat_labels,
            branch_names,
            "Average routing weights by relation type",
            os.path.join(args.out_dir, "routing_relation_type_wn18rr")
        )

    case_rel_ids = select_case_relations(rel_names, rel_ids_for_fig, alpha, args.case_relations)
    case_rows = []
    case_values = []
    case_labels = []
    rel_to_idx = {int(r): i for i, r in enumerate(rel_ids_for_fig)}
    for r in case_rel_ids:
        if r not in rel_to_idx:
            continue
        idx = rel_to_idx[r]
        row = {
            "rel_id": int(r),
            "relation": rel_names.get(int(r), f"rel_{r}"),
            "relation_group": rel_group[int(r)],
        }
        for j in range(alpha.shape[1]):
            row[f"alpha_{j+1}"] = float(alpha[idx, j])
        case_rows.append(row)
        case_values.append(alpha[idx])
        case_labels.append(rel_names.get(int(r), f"rel_{r}").replace("_", "\\_"))
    if case_rows:
        write_csv(
            os.path.join(args.out_dir, "routing_cases.csv"),
            case_rows,
            ["rel_id", "relation", "relation_group"] + [f"alpha_{j+1}" for j in range(alpha.shape[1])]
        )
        plot_grouped_bars(
            case_labels,
            np.asarray(case_values, dtype=np.float32),
            branch_names,
            "Case-level routing behavior",
            "Routing weight",
            os.path.join(args.out_dir, "routing_case_wn18rr")
        )

    # ------------------------- 4.5 Evidence analysis -------------------------
    gates, effective_gates = extract_evidence_gates(model, rel_ids_for_fig, device)
    gate_rows = []
    for idx, r in enumerate(rel_ids_for_fig):
        gate_rows.append({
            "rel_id": int(r),
            "relation": rel_names.get(int(r), f"rel_{int(r)}"),
            "relation_group": rel_group[int(r)],
            "base_gate": float(gates[idx, 0]),
            "path_gate": float(gates[idx, 1]),
            "role_gate": float(gates[idx, 2]),
            "effective_path_strength": float(effective_gates[idx, 0]),
            "effective_role_strength": float(effective_gates[idx, 1]),
        })
    write_csv(
        os.path.join(args.out_dir, "evidence_gates_by_relation.csv"),
        gate_rows,
        ["rel_id", "relation", "relation_group", "base_gate", "path_gate", "role_gate",
         "effective_path_strength", "effective_role_strength"]
    )

    gate_group_rows, gate_group_fields = summarize_by_group(gates, groups_for_fig, "gate_")
    # Rename gate_1/gate_2/gate_3 to base/path/role for readability.
    renamed_gate_rows = []
    for row in gate_group_rows:
        renamed_gate_rows.append({
            "relation_group": row["relation_group"],
            "count": row["count"],
            "base_gate": row.get("gate_1", np.nan),
            "path_gate": row.get("gate_2", np.nan),
            "role_gate": row.get("gate_3", np.nan),
        })
    write_csv(
        os.path.join(args.out_dir, "evidence_gates_by_type.csv"),
        renamed_gate_rows,
        ["relation_group", "count", "base_gate", "path_gate", "role_gate"]
    )
    if renamed_gate_rows:
        labels = [r["relation_group"] for r in renamed_gate_rows]
        values = np.asarray([[r["base_gate"], r["path_gate"], r["role_gate"]] for r in renamed_gate_rows], dtype=np.float32)
        plot_grouped_bars(
            labels,
            values,
            ["Base gate", "Path gate", "Role gate"],
            "Average evidence gates by relation type",
            "Gate value",
            os.path.join(args.out_dir, "evidence_gate_wn18rr")
        )

    evidence_rows = evaluate_evidence_variants(
        dataset, model, split=args.split, batch_size=args.batch_size, device=device,
        rel_group=rel_group, directions=args.directions
    )
    write_csv(
        os.path.join(args.out_dir, "evidence_effect_by_type.csv"),
        evidence_rows,
        ["relation_group", "Full RCAR", "w/o path", "w/o role", "w/o both",
         "Full RCAR_count", "w/o path_count", "w/o role_count", "w/o both_count"]
    )

    # Plot MRR table for evidence variants.
    plot_labels = [r["relation_group"] for r in evidence_rows]
    plot_values = np.asarray([[r["Full RCAR"], r["w/o path"], r["w/o role"], r["w/o both"]] for r in evidence_rows], dtype=np.float32)
    plot_grouped_bars(
        plot_labels,
        plot_values,
        ["Full RCAR", "w/o path", "w/o role", "w/o both"],
        "Effect of path and role evidence",
        "MRR",
        os.path.join(args.out_dir, "evidence_effect_by_type_wn18rr")
    )

    # -------------------------- 4.5.3 Case study -----------------------------
    case_study_rows = collect_case_study(
        dataset, model, split=args.split, batch_size=args.batch_size, device=device,
        rel_group=rel_group, rel_names=rel_names, ent_names=ent_names,
        directions=args.directions, max_cases=args.max_cases
    )
    if case_study_rows:
        write_csv(
            os.path.join(args.out_dir, "qualitative_cases.csv"),
            case_study_rows,
            ["direction", "head_id", "head", "relation_id", "relation", "tail_id", "target",
             "relation_group", "base_rank", "full_rank", "rank_gain",
             "rank_without_path", "rank_without_role", "main_evidence"]
        )

    print("\nDone. Analysis files saved to:", os.path.abspath(args.out_dir))
    print("Main outputs:")
    for name in [
        "routing_by_type.csv",
        "routing_by_relation.csv",
        "routing_cases.csv",
        "evidence_gates_by_type.csv",
        "evidence_gates_by_relation.csv",
        "evidence_effect_by_type.csv",
        "qualitative_cases.csv",
        "routing_relation_type_wn18rr.pdf",
        "routing_case_wn18rr.pdf",
        "evidence_gate_wn18rr.pdf",
        "evidence_effect_by_type_wn18rr.pdf",
    ]:
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p):
            print("  -", p)


if __name__ == "__main__":
    main()
