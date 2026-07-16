import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import torch
from collections import defaultdict, Counter
from models import KBCModel


class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        # 通过三个pickle文件加载数据集
        # 就是转化为id后的三元组
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)
        # self.data['train'] 的内容举例:[[0,0,2], [0,1,3], [1,0,2], [2,2,4]]

        print(self.data['train'].shape)
        # [[0,0,2], [0,1,3], [1,0,2], [2,2,4]]
        # [2, 2, 4]
        # 计算每一列的最大值
        maxis = np.max(self.data['train'], axis=0)
        # [[0,0,2], [0,1,3], [1,0,2], [2,2,4]] maxis = [2, 2, 4]
        # n_entities 实体的适量
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        # 关系的数量
        self.n_predicates = int(maxis[1] + 1)
        # 真实的关系数量 (不包括反向关系)
        self.real_r = self.n_predicates
        self.relaions = []
        self.MRR = []
        self.hit1 = []

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        # to_skip的结构如下:{''lhs': {(rhs_id, rel_id): [lhs_id1, lhs_id2, ...], ...}, 'rhs': {(lhs_id, rel_id): [rhs_id1, rhs_id2, ...], ...}}

    # 计算测试集中每个实体的出现权重，用于加权评估。
    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['test'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1
        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        # 返回的结果的结构为：[w_0, w_1, w_2, ..., w_n] n为实体数量
        return w

    def get_weight_from_split(self, split='train'):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data[split])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1
        max_count = np.max(appear_list)
        if max_count <= 0:
            return np.ones(self.n_entities)
        return appear_list / max_count * 0.9 + 0.1

    # 获取每个实体的原始出现频率
    def get_frequencies(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1
        return appear_list

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        # 关系变为了2倍
        # 建立反向关系
        sdata = self.data['train']
        r_c = self.real_r
        copy = np.copy(sdata)
        # 就是将头尾实体交换，并且关系加上 r_c（翻转关系）
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += r_c
        self.n_predicates = r_c*2
        # np.vstack 是 NumPy 的垂直堆叠函数，将多个数组按行（垂直方向拼接在一起。
        sdata = np.vstack((sdata, copy))
        self.train_with_reciprocals = sdata
        # 三元组的数量变为了2倍
        return sdata

    def get_multi_positive_train(self):
        """
        Group reciprocal training triples by query ``(head, relation)``.

        The returned query tensor keeps one representative tail in its third
        column so existing model forward/regularizer interfaces remain
        compatible. ``positive_targets`` contains every known training tail
        for the corresponding query and is the actual supervision used by the
        multi-positive softmax loss.
        """
        train = self._ensure_train_with_reciprocals().astype('int64')
        tails_by_query = defaultdict(set)
        for h, r, t in train:
            tails_by_query[(int(h), int(r))].add(int(t))

        query_keys = sorted(tails_by_query)
        queries = np.zeros((len(query_keys), 3), dtype=np.int64)
        positive_targets = []
        for i, (h, r) in enumerate(query_keys):
            tails = np.asarray(
                sorted(tails_by_query[(h, r)]), dtype=np.int64
            )
            queries[i] = (h, r, int(tails[0]))
            positive_targets.append(torch.from_numpy(tails))

        return torch.from_numpy(queries), positive_targets

    def _ensure_train_with_reciprocals(self):
        if not hasattr(self, 'train_with_reciprocals'):
            self.get_train()
        return self.train_with_reciprocals

    def build_relation_context(self, include_entity_stats=True):
        """
        Build relation and entity structural statistics from training triples only.

        Relation statistics use reciprocal triples because the model predicts both
        directions. Entity statistics use the original directed training graph so
        incoming and outgoing roles remain distinguishable.
        """
        train = self._ensure_train_with_reciprocals().astype('int64')
        n_ent = self.n_entities
        n_rel = self.n_predicates

        rel_count = np.zeros(n_rel, dtype=np.float32)
        heads_by_rel = [set() for _ in range(n_rel)]
        tails_by_rel = [set() for _ in range(n_rel)]
        hr_tails = defaultdict(set)
        rt_heads = defaultdict(set)
        tail_counter_by_rel = [Counter() for _ in range(n_rel)]
        head_counter_by_rel = [Counter() for _ in range(n_rel)]

        for h, r, t in train:
            h = int(h); r = int(r); t = int(t)
            rel_count[r] += 1
            heads_by_rel[r].add(h)
            tails_by_rel[r].add(t)
            hr_tails[(h, r)].add(t)
            rt_heads[(t, r)].add(h)
            tail_counter_by_rel[r][t] += 1
            head_counter_by_rel[r][h] += 1

        rel_stats = np.zeros((n_rel, 10), dtype=np.float32)
        max_freq = max(float(np.max(rel_count)), 1.0)
        tph_values, hpt_values = [], []
        raw_stats = []
        for r in range(n_rel):
            tails_per_head = [len(hr_tails[(h, r)]) for h in heads_by_rel[r]]
            heads_per_tail = [len(rt_heads[(t, r)]) for t in tails_by_rel[r]]
            tph = float(np.mean(tails_per_head)) if tails_per_head else 0.0
            hpt = float(np.mean(heads_per_tail)) if heads_per_tail else 0.0
            tph_values.append(tph)
            hpt_values.append(hpt)
            raw_stats.append((tph, hpt))

        max_tph = max(max(tph_values), 1.0)
        max_hpt = max(max(hpt_values), 1.0)
        log_ent = np.log(max(n_ent, 2))
        for r in range(n_rel):
            tph, hpt = raw_stats[r]
            rel_stats[r, 0] = np.log1p(rel_count[r]) / np.log1p(max_freq)
            rel_stats[r, 1] = np.log1p(tph) / np.log1p(max_tph)
            rel_stats[r, 2] = np.log1p(hpt) / np.log1p(max_hpt)

            tail_total = sum(tail_counter_by_rel[r].values())
            if tail_total > 0:
                probs = np.array(list(tail_counter_by_rel[r].values()), dtype=np.float32) / tail_total
                rel_stats[r, 3] = float(-np.sum(probs * np.log(probs + 1e-12)) / log_ent)
            head_total = sum(head_counter_by_rel[r].values())
            if head_total > 0:
                probs = np.array(list(head_counter_by_rel[r].values()), dtype=np.float32) / head_total
                rel_stats[r, 4] = float(-np.sum(probs * np.log(probs + 1e-12)) / log_ent)

            if tph < 1.5 and hpt < 1.5:
                rel_type = 0
            elif tph >= 1.5 and hpt < 1.5:
                rel_type = 1
            elif tph < 1.5 and hpt >= 1.5:
                rel_type = 2
            else:
                rel_type = 3
            rel_stats[r, 5 + rel_type] = 1.0
            rel_stats[r, 9] = 1.0 if r >= self.real_r else 0.0

        context = {
            'rel_stats': torch.from_numpy(rel_stats),
        }

        if include_entity_stats:
            directed_train = self.data['train'].astype('int64')
            heads = directed_train[:, 0]
            rels = directed_train[:, 1]
            tails = directed_train[:, 2]

            out_degree = np.bincount(heads, minlength=n_ent).astype(np.float32)
            in_degree = np.bincount(tails, minlength=n_ent).astype(np.float32)

            def role_summary(entity_ids, relation_ids, degree):
                # Encode (entity, relation) pairs without constructing a dense
                # n_entity x n_relation matrix.
                pair_ids = entity_ids * self.real_r + relation_ids
                unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)
                pair_entities = unique_pairs // self.real_r

                diversity = np.bincount(
                    pair_entities, minlength=n_ent
                ).astype(np.float32)
                diversity /= max(float(self.real_r), 1.0)

                pair_counts = pair_counts.astype(np.float32)
                xlogx = pair_counts * np.log(np.maximum(pair_counts, 1.0))
                summed_xlogx = np.bincount(
                    pair_entities, weights=xlogx, minlength=n_ent
                ).astype(np.float32)

                entropy = np.zeros(n_ent, dtype=np.float32)
                active = degree > 0
                entropy[active] = (
                    np.log(degree[active])
                    - summed_xlogx[active] / degree[active]
                ) / np.log(max(self.real_r, 2))
                return diversity, np.clip(entropy, 0.0, 1.0)

            out_diversity, out_entropy = role_summary(heads, rels, out_degree)
            in_diversity, in_entropy = role_summary(tails, rels, in_degree)

            def normalized_log_degree(degree):
                scale = np.log1p(max(float(degree.max()), 1.0))
                return np.log1p(degree) / scale

            total_degree = out_degree + in_degree
            role_balance = (out_degree - in_degree) / np.maximum(total_degree, 1.0)
            entity_stats = np.stack([
                normalized_log_degree(out_degree),
                normalized_log_degree(in_degree),
                out_diversity,
                in_diversity,
                out_entropy,
                in_entropy,
                normalized_log_degree(total_degree),
                role_balance,
            ], axis=1).astype(np.float32)
            context['entity_stats'] = torch.from_numpy(entity_stats)

        return context

    def build_rcem_context(
            self,
            max_rules_per_relation=8,
            max_candidates_per_query=32,
            min_rule_support=3,
            max_rule_degree=64,
            standard_confidence_weight=0.3,
            rule_smoothing=1.0,
            use_path=True,
            use_type=True,
    ):
        """
        Build relation-conditioned evidence memory from training triples only.

        Path evidence mines two-hop relation compositions:
            r1(h, z) and r2(z, t) -> r(h, t)
        Type evidence uses unsupervised entity role signatures formed by
        incoming/outgoing relation distributions.
        """
        train = self._ensure_train_with_reciprocals().astype('int64')
        n_ent = self.n_entities
        n_rel = self.n_predicates
        context = {}

        if use_path:
            max_rules = max(1, int(max_rules_per_relation))
            max_candidates = max(1, int(max_candidates_per_query))
            min_support = max(1, int(min_rule_support))
            max_degree = max(1, int(max_rule_degree))
            standard_weight = min(max(float(standard_confidence_weight), 0.0), 1.0)
            smoothing = max(float(rule_smoothing), 1e-6)

            out_by_head = defaultdict(list)
            in_by_tail = defaultdict(list)
            out_by_rel = [defaultdict(list) for _ in range(n_rel)]
            rel_count = np.zeros(n_rel, dtype=np.float32)

            for h, r, t in train:
                h = int(h); r = int(r); t = int(t)
                out_by_head[h].append((r, t))
                in_by_tail[t].append((h, r))
                out_by_rel[r][h].append(t)
                rel_count[r] += 1

            for h in out_by_head:
                out_by_head[h] = sorted(out_by_head[h], key=lambda x: (x[0], x[1]))[:max_degree]
            for t in in_by_tail:
                in_by_tail[t] = sorted(in_by_tail[t], key=lambda x: (x[0], x[1]))[:max_degree]
            for r in range(n_rel):
                for h in out_by_rel[r]:
                    out_by_rel[r][h] = sorted(set(out_by_rel[r][h]))[:max_degree]

            # Count positive (h, t) facts explained by each rule. A rule
            # receives at most one support vote per target triple even if
            # several intermediate entities instantiate the same body.
            rule_support = [Counter() for _ in range(n_rel)]
            for h, r, t in train:
                h = int(h); r = int(r); t = int(t)
                left_edges = out_by_head.get(h, [])
                right_edges = in_by_tail.get(t, [])
                if not left_edges or not right_edges:
                    continue

                left_by_mid = defaultdict(list)
                for r1, mid in left_edges:
                    left_by_mid[mid].append(r1)

                matched_rules = set()
                for mid, r2 in right_edges:
                    if mid not in left_by_mid:
                        continue
                    for r1 in left_by_mid[mid]:
                        matched_rules.add((r1, r2))
                for rule in matched_rules:
                    rule_support[r][rule] += 1

            # Preselect by positive support, then compute exact distinct body
            # support only for a small candidate pool. This keeps confidence
            # estimation bounded while avoiding a full all-rule enumeration.
            targets_by_body = defaultdict(list)
            preselect_limit = max_rules * 4
            for r in range(n_rel):
                candidates = []
                for (r1, r2), support in rule_support[r].items():
                    if support < min_support:
                        continue
                    head_coverage = support / max(float(rel_count[r]), 1.0)
                    candidates.append((head_coverage, support, r1, r2))
                candidates.sort(reverse=True)
                candidates = candidates[:preselect_limit]
                for _, support, r1, r2 in candidates:
                    targets_by_body[(r1, r2)].append((r, support))

            # Enumerate candidate rule bodies in one graph pass. Re-running a
            # full graph traversal for every rule is substantially slower on
            # datasets with many relations.
            candidate_bodies = set(targets_by_body)
            body_support_by_rule = Counter()
            pca_support_by_rule = Counter()
            for h, first_edges in out_by_head.items():
                local_body_tails = defaultdict(set)
                for r1, mid in first_edges:
                    for r2, tail in out_by_head.get(mid, []):
                        body = (r1, r2)
                        if body in candidate_bodies:
                            local_body_tails[body].add(tail)

                for body, body_tails in local_body_tails.items():
                    count = len(body_tails)
                    body_support_by_rule[body] += count
                    for target_r, _ in targets_by_body[body]:
                        if h in out_by_rel[target_r]:
                            pca_support_by_rule[(body, target_r)] += count

            scored_by_rel = [[] for _ in range(n_rel)]
            for (r1, r2), target_rules in targets_by_body.items():
                body = (r1, r2)
                body_support = body_support_by_rule[body]
                for target_r, support in target_rules:
                    pca_body_support = pca_support_by_rule[(body, target_r)]
                    # Denominators are guarded by support because the bounded
                    # adjacency views used by positive and body passes differ.
                    standard_denom = max(float(body_support), float(support))
                    pca_denom = max(float(pca_body_support), float(support))
                    standard_confidence = (
                        float(support) + smoothing
                    ) / (standard_denom + 2.0 * smoothing)
                    pca_confidence = (
                        float(support) + smoothing
                    ) / (pca_denom + 2.0 * smoothing)
                    head_coverage = min(
                        float(support) / max(float(rel_count[target_r]), 1.0),
                        1.0,
                    )
                    confidence = (
                        standard_weight * standard_confidence
                        + (1.0 - standard_weight) * pca_confidence
                    )
                    quality = (
                        np.log1p(float(support))
                        * np.sqrt(max(head_coverage, 1e-12))
                        * confidence
                    )
                    scored_by_rel[target_r].append((
                        quality, r1, r2, support,
                        standard_confidence, pca_confidence, head_coverage,
                    ))

            rules_by_rel = []
            for scored_rules in scored_by_rel:
                scored_rules.sort(reverse=True)
                rules_by_rel.append(scored_rules[:max_rules])

            evidence_scores = defaultdict(dict)
            prune_limit = max_candidates * 4
            keep_limit = max_candidates * 2

            for target_r, rules in enumerate(rules_by_rel):
                if not rules:
                    continue
                for score, r1, r2, _, _, _, _ in rules:
                    first_hops = out_by_rel[r1]
                    if not first_hops:
                        continue
                    for h, mids in first_hops.items():
                        rule_tail_counts = Counter()
                        for mid in mids:
                            tails = out_by_rel[r2].get(mid, [])
                            for tail in tails:
                                rule_tail_counts[tail] += 1
                        if not rule_tail_counts:
                            continue
                        q_scores = evidence_scores[(h, target_r)]
                        for tail, path_count in rule_tail_counts.items():
                            q_scores[tail] = (
                                q_scores.get(tail, 0.0)
                                + float(score) * np.log1p(float(path_count))
                            )
                        if len(q_scores) > prune_limit:
                            top_items = sorted(q_scores.items(), key=lambda x: x[1], reverse=True)[:keep_limit]
                            evidence_scores[(h, target_r)] = dict(top_items)

            # Relation-level calibration keeps evidence comparable across
            # queries while bounding additions to [0, 1].
            relation_score_sum = np.zeros(n_rel, dtype=np.float64)
            relation_score_count = np.zeros(n_rel, dtype=np.int64)
            for (_, r), scores in evidence_scores.items():
                relation_score_sum[r] += sum(scores.values())
                relation_score_count[r] += len(scores)
            relation_score_scale = relation_score_sum / np.maximum(
                relation_score_count, 1
            )

            query_index = np.full((n_ent, n_rel), -1, dtype=np.int64)
            num_queries = len(evidence_scores)
            candidate_ids = np.zeros((num_queries, max_candidates), dtype=np.int64)
            candidate_scores = np.zeros((num_queries, max_candidates), dtype=np.float32)
            query_features = np.zeros((num_queries, 6), dtype=np.float32)

            for idx, ((h, r), scores) in enumerate(evidence_scores.items()):
                query_index[h, r] = idx
                top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
                if not top_items:
                    continue
                raw_values = np.array([v for _, v in top_items], dtype=np.float32)
                scale = max(float(relation_score_scale[r]), 1e-6)
                values = 1.0 - np.exp(-raw_values / scale)
                candidate_ids[idx, :len(top_items)] = [t for t, _ in top_items]
                candidate_scores[idx, :len(top_items)] = values

                value_sum = max(float(values.sum()), 1e-8)
                probabilities = values / value_sum
                if len(values) > 1:
                    entropy = -float(np.sum(
                        probabilities * np.log(probabilities + 1e-12)
                    )) / np.log(float(len(values)))
                    concentration = 1.0 - entropy
                    margin = float(values[0] - values[1])
                else:
                    concentration = 1.0
                    margin = float(values[0])
                query_features[idx] = [
                    1.0,
                    len(values) / float(max_candidates),
                    float(values[0]),
                    float(values.mean()),
                    margin,
                    concentration,
                ]

            context['path_query_index'] = torch.from_numpy(query_index)
            context['path_candidate_ids'] = torch.from_numpy(candidate_ids)
            context['path_candidate_scores'] = torch.from_numpy(candidate_scores)
            context['path_query_features'] = torch.from_numpy(query_features)

        if use_type:
            role_dim = n_rel * 2
            role = np.zeros((n_ent, role_dim), dtype=np.float32)
            tail_proto = np.zeros((n_rel, role_dim), dtype=np.float32)
            tail_count = np.zeros(n_rel, dtype=np.float32)

            for h, r, t in train:
                h = int(h); r = int(r); t = int(t)
                role[h, r] += 1.0
                role[t, n_rel + r] += 1.0

            role = np.log1p(role)
            role_norm = np.linalg.norm(role, axis=1, keepdims=True)
            role = role / np.maximum(role_norm, 1e-6)

            for h, r, t in train:
                r = int(r); t = int(t)
                tail_proto[r] += role[t]
                tail_count[r] += 1.0

            tail_proto = tail_proto / np.maximum(tail_count[:, None], 1.0)
            proto_norm = np.linalg.norm(tail_proto, axis=1, keepdims=True)
            tail_proto = tail_proto / np.maximum(proto_norm, 1e-6)

            type_scores = np.matmul(tail_proto, role.T).astype(np.float32)
            for r in range(n_rel):
                row = type_scores[r]
                row_min = float(row.min())
                row_max = float(row.max())
                if row_max > row_min:
                    type_scores[r] = (row - row_min) / (row_max - row_min)
                else:
                    type_scores[r] = 0.0

            context['type_scores'] = torch.from_numpy(type_scores)

            # Reliability descriptors for query-aware gating only. The type
            # evidence construction itself intentionally remains unchanged.
            encoded_rel_tails = train[:, 1] * n_ent + train[:, 2]
            unique_rel_tails = np.unique(encoded_rel_tails)
            unique_tail_rels = unique_rel_tails // n_ent
            unique_tail_ids = unique_rel_tails % n_ent
            unique_tail_count = np.bincount(
                unique_tail_rels, minlength=n_rel
            ).astype(np.float32)
            cohesion_sum = np.bincount(
                unique_tail_rels,
                weights=type_scores[unique_tail_rels, unique_tail_ids],
                minlength=n_rel,
            ).astype(np.float32)
            cohesion = cohesion_sum / np.maximum(unique_tail_count, 1.0)

            support_scale = np.log1p(max(float(tail_count.max()), 1.0))
            unique_scale = np.log1p(max(float(unique_tail_count.max()), 1.0))
            type_reliability = np.stack([
                np.log1p(tail_count) / support_scale,
                np.log1p(unique_tail_count) / unique_scale,
                cohesion,
                np.std(type_scores, axis=1),
            ], axis=1).astype(np.float32)
            context['type_reliability'] = torch.from_numpy(type_reliability)

        return context

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        # 评估模型在指定数据集上的表现
        model.eval()
        # 得到数据集
        test = self.get_examples(split)
        # 将数据放在cuda中
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        # 评估的的模式，是预测头实体还是尾实体，还是两者都评估
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            # 这里的examples是数据集中的三元组结构为[[lhs_id, rel_id, rhs_id], ...],为所有的测试集的数据
            # n_queries 表示要评估的查询数量，-1表示评估所有查询，默认为-1
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
                
            if m == 'lhs':
                # 预测头实体
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.real_r
            # ranks 是一个长度为 n_queries 的一维数组，表示每个查询的正确实体在排序中的位置
            # 意思就是测试了这么多，每一个测试的三元组，正确的实体在所有实体中的排名是多少[1,2,3,9,20,...]
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    # examples为数据集中的三元组结构为[[lhs_id, rel_id, rhs_id], ...]
    def get_pos(self, examples):
        # 这个pos表示的是positive samples的意思
        dic_tr ={}
        dic_hr = {}
        dic_t = {}
        dic_h = {}
        for i in examples:
            # tail-relation: 给定头实体，找所有 (尾, 关系)
            # 这些(尾, 关系)共享同一个头实体
            # .item() 是 PyTorch 的方法，用于将只包含一个元素的张量转换为 Python 标量
            dic_tr[i[0].item()] = []
            # head-relation: 给定尾实体，找所有 (头, 关系)
            # 这些(头, 关系)共享同一个尾实体
            dic_hr[i[2].item()] = []
            # tail: 给定 (头, 关系)，找所有尾实体，这些尾实体共享同一个(头, 关系)
            dic_t[(i[0].item(), i[1].item())] =[]
            # head: 给定 (尾, 关系)，找所有头，这些头共享同一个(尾, 关系)
            dic_h[(i[2].item(), i[1].item())] = []
        for i in examples:
            dic_tr[i[0].item()].append([i[2].item(), i[1].item()])
            dic_hr[i[2].item()].append([i[0].item(), i[1].item()])
            dic_t[(i[0].item(), i[1].item())].append(i[2].item())
            dic_h[(i[2].item(), i[1].item())].append(i[0].item())
        return dic_tr, dic_hr, dic_h, dic_t

    def get_shape(self):
        # 得到实体数量，关系数量，实体数量
        return self.n_entities, self.n_predicates, self.n_entities
