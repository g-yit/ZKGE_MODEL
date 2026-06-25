import math
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from models import KBCModel


class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)
        self.data = {}
        for split in ["train", "test", "valid"]:
            with open(os.path.join(self.root, split + ".pickle"), "rb") as in_file:
                self.data[split] = pickle.load(in_file)

        all_examples = np.vstack([self.data["train"], self.data["valid"], self.data["test"]])
        maxis = np.max(all_examples, axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.real_r = self.n_predicates

        with open(os.path.join(self.root, "to_skip.pickle"), "rb") as inp_f:
            self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)

        self._train_with_reciprocals = None
        self._positive_dict = None
        self._relation_anchor_pools = None
        self._relation_stats = None

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    def get_examples(self, split):
        return self.data[split]

    def get_train(self, add_reciprocal: bool = True):
        train = self.data["train"].astype("int64")
        if not add_reciprocal:
            self._build_training_cache(train, self.real_r)
            return train

        copy = np.copy(train)
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.real_r
        self.n_predicates = self.real_r * 2
        train_with_inv = np.vstack((train, copy)).astype("int64")
        self._build_training_cache(train_with_inv, self.n_predicates)
        return train_with_inv

    def get_weight(self):
        train = self._train_with_reciprocals
        if train is None:
            train = self.data["train"].astype("int64")
        appear = np.zeros(self.n_entities, dtype=np.float32)
        for h, _, t in train:
            appear[h] += 1.0
            appear[t] += 1.0
        if appear.max() <= 0:
            return np.ones(self.n_entities, dtype=np.float32)
        return appear / appear.max() * 0.9 + 0.1

    def get_frequencies(self):
        appear = np.zeros(self.n_entities, dtype=np.float32)
        for h, _, t in self.data["train"].astype("int64"):
            appear[h] += 1.0
            appear[t] += 1.0
        return appear

    def _flat_key(self, h: int, r: int) -> int:
        return int(h) * self.n_predicates + int(r)

    def _build_training_cache(self, examples: np.ndarray, num_rel: int):
        self._train_with_reciprocals = examples.astype("int64")
        self.n_predicates = num_rel

        positive_sets = defaultdict(set)
        relation_tail_counter = [Counter() for _ in range(num_rel)]
        rel_heads = [set() for _ in range(num_rel)]
        rel_tails = [set() for _ in range(num_rel)]
        head_counter = [Counter() for _ in range(num_rel)]
        tail_counter = [Counter() for _ in range(num_rel)]
        rel_count = np.zeros(num_rel, dtype=np.float32)

        for h, r, t in self._train_with_reciprocals:
            h, r, t = int(h), int(r), int(t)
            positive_sets[self._flat_key(h, r)].add(t)
            relation_tail_counter[r][t] += 1
            rel_heads[r].add(h)
            rel_tails[r].add(t)
            head_counter[r][h] += 1
            tail_counter[r][t] += 1
            rel_count[r] += 1.0

        self._positive_dict = {k: sorted(v) for k, v in positive_sets.items()}
        self._relation_anchor_pools = [
            [ent for ent, _ in counter.most_common(256)]
            for counter in relation_tail_counter
        ]
        self._relation_stats = self._make_relation_stats(
            rel_count, rel_heads, rel_tails, head_counter, tail_counter
        )

    def _entropy(self, counter: Counter, denom: float) -> float:
        if denom <= 0 or not counter:
            return 0.0
        ent = 0.0
        for count in counter.values():
            p = count / denom
            ent -= p * math.log(max(p, 1e-12))
        return ent

    def _make_relation_stats(
        self,
        rel_count: np.ndarray,
        rel_heads: List[set],
        rel_tails: List[set],
        head_counter: List[Counter],
        tail_counter: List[Counter],
    ):
        stats = np.zeros((len(rel_count), 8), dtype=np.float32)
        for r, count in enumerate(rel_count):
            heads = max(len(rel_heads[r]), 1)
            tails = max(len(rel_tails[r]), 1)
            tph = count / heads if count > 0 else 0.0
            hpt = count / tails if count > 0 else 0.0
            head_ent = self._entropy(head_counter[r], float(count))
            tail_ent = self._entropy(tail_counter[r], float(count))
            head_ent /= math.log(heads + 1.0)
            tail_ent /= math.log(tails + 1.0)
            stats[r] = np.array(
                [
                    math.log1p(count),
                    math.log1p(heads),
                    math.log1p(tails),
                    math.log1p(tph),
                    math.log1p(hpt),
                    1.0 / max(tph, 1e-6),
                    1.0 / max(hpt, 1e-6),
                    0.5 * (head_ent + tail_ent),
                ],
                dtype=np.float32,
            )
        mean = stats.mean(axis=0, keepdims=True)
        std = stats.std(axis=0, keepdims=True) + 1e-6
        return (stats - mean) / std

    def get_relation_stats(self):
        if self._relation_stats is None:
            self.get_train(add_reciprocal=True)
        return self._relation_stats.astype("float32")

    def get_positive_batch(
        self,
        batch: torch.Tensor,
        max_positives: int = 64,
    ):
        if self._positive_dict is None:
            self.get_train(add_reciprocal=True)

        rows, cols, counts = [], [], []
        batch_np = batch.detach().cpu().numpy()
        for row, (h, r, t) in enumerate(batch_np):
            positives = self._positive_dict.get(self._flat_key(int(h), int(r)), [int(t)])
            if max_positives and len(positives) > max_positives:
                if int(t) in positives:
                    rest = [p for p in positives if p != int(t)]
                    sampled = rest[: max_positives - 1]
                    positives = [int(t)] + sampled
                else:
                    positives = positives[:max_positives]
            counts.append(max(len(positives), 1))
            rows.extend([row] * len(positives))
            cols.extend(positives)

        return (
            torch.as_tensor(rows, dtype=torch.long),
            torch.as_tensor(cols, dtype=torch.long),
            torch.as_tensor(counts, dtype=torch.float32),
        )

    def get_anchor_batch(
        self,
        batch: torch.Tensor,
        topk: int,
        device: torch.device,
        exclude_target: bool = True,
    ):
        if topk <= 0:
            bsz = batch.shape[0]
            return (
                torch.empty((bsz, 0), dtype=torch.long, device=device),
                torch.empty((bsz, 0), dtype=torch.bool, device=device),
            )
        if self._positive_dict is None:
            self.get_train(add_reciprocal=True)

        batch_np = batch.detach().cpu().numpy()
        anchors = np.full((len(batch_np), topk), -1, dtype=np.int64)
        mask = np.zeros((len(batch_np), topk), dtype=np.bool_)

        for row, (h, r, t) in enumerate(batch_np):
            h, r, t = int(h), int(r), int(t)
            selected = []
            for ent in self._positive_dict.get(self._flat_key(h, r), []):
                if exclude_target and ent == t:
                    continue
                selected.append(ent)
                if len(selected) >= topk:
                    break

            if len(selected) < topk and r < len(self._relation_anchor_pools):
                used = set(selected)
                for ent in self._relation_anchor_pools[r]:
                    if exclude_target and ent == t:
                        continue
                    if ent in used:
                        continue
                    selected.append(ent)
                    used.add(ent)
                    if len(selected) >= topk:
                        break

            if selected:
                anchors[row, : len(selected)] = selected
                mask[row, : len(selected)] = True

        return (
            torch.as_tensor(anchors, dtype=torch.long, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
        )

    def eval(
        self,
        model: KBCModel,
        split: str,
        n_queries: int = -1,
        missing_eval: str = "both",
        at: Tuple[int] = (1, 3, 10),
        batch_size: int = 500,
    ):
        model.eval()
        examples = torch.from_numpy(self.get_examples(split).astype("int64")).to(model.device)
        missing = ["rhs", "lhs"] if missing_eval == "both" else [missing_eval]

        mean_reciprocal_rank = {}
        hits_at = {}

        for side in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples), device=examples.device)[:n_queries]
                q = examples[permutation].clone()

            if side == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.real_r

            ranks = model.get_ranking(q, self.to_skip[side], batch_size=batch_size)
            mean_reciprocal_rank[side] = torch.mean(1.0 / ranks).item()
            hits_at[side] = torch.FloatTensor(
                [torch.mean((ranks <= k).float()).item() for k in at]
            )

        return mean_reciprocal_rank, hits_at
