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

    def _ensure_train_with_reciprocals(self):
        if not hasattr(self, 'train_with_reciprocals'):
            self.get_train()
        return self.train_with_reciprocals

    def build_graph_context(self, max_neighbors=16, min_pmi_count=2):
        """
        Build fixed-size graph context tensors from training triples only.

        The context contains:
        - relation-balanced neighbors for each entity;
        - relation-pair PMI measured from local relation co-occurrence;
        - relation cardinality/frequency/entropy/type statistics.
        """
        train = self._ensure_train_with_reciprocals().astype('int64')
        n_ent = self.n_entities
        n_rel = self.n_predicates

        neighbors = [[] for _ in range(n_ent)]
        degree = np.zeros(n_ent, dtype=np.float32)
        rel_count = np.zeros(n_rel, dtype=np.float32)
        heads_by_rel = [set() for _ in range(n_rel)]
        tails_by_rel = [set() for _ in range(n_rel)]
        hr_tails = defaultdict(set)
        rt_heads = defaultdict(set)
        tail_counter_by_rel = [Counter() for _ in range(n_rel)]
        head_counter_by_rel = [Counter() for _ in range(n_rel)]
        rels_by_entity = [set() for _ in range(n_ent)]

        for h, r, t in train:
            h = int(h); r = int(r); t = int(t)
            neighbors[h].append((r, t))
            degree[h] += 1
            degree[t] += 1
            rel_count[r] += 1
            heads_by_rel[r].add(h)
            tails_by_rel[r].add(t)
            hr_tails[(h, r)].add(t)
            rt_heads[(t, r)].add(h)
            tail_counter_by_rel[r][t] += 1
            head_counter_by_rel[r][h] += 1
            rels_by_entity[h].add(r)

        neighbor_entities = np.zeros((n_ent, max_neighbors), dtype=np.int64)
        neighbor_relations = np.zeros((n_ent, max_neighbors), dtype=np.int64)
        neighbor_mask = np.zeros((n_ent, max_neighbors), dtype=np.float32)

        for ent, cand in enumerate(neighbors):
            if not cand:
                continue
            by_rel = defaultdict(list)
            for r, t in cand:
                by_rel[r].append((r, t))
            quota = max(1, max_neighbors // max(1, len(by_rel)))
            selected = []
            used = set()
            for r in sorted(by_rel.keys(), key=lambda x: -rel_count[x]):
                rel_cand = sorted(by_rel[r], key=lambda item: (degree[item[1]], item[1]))
                for item in rel_cand[:quota]:
                    if len(selected) >= max_neighbors:
                        break
                    selected.append(item)
                    used.add(item)
                if len(selected) >= max_neighbors:
                    break
            if len(selected) < max_neighbors:
                remaining = [item for item in cand if item not in used]
                remaining = sorted(remaining, key=lambda item: (degree[item[1]], -rel_count[item[0]], item[1]))
                selected.extend(remaining[:max_neighbors - len(selected)])

            for j, (r, t) in enumerate(selected[:max_neighbors]):
                neighbor_entities[ent, j] = t
                neighbor_relations[ent, j] = r
                neighbor_mask[ent, j] = 1.0

        rel_cooc = np.zeros((n_rel, n_rel), dtype=np.float32)
        rel_entity_count = np.zeros(n_rel, dtype=np.float32)
        for rels in rels_by_entity:
            if not rels:
                continue
            rel_list = list(rels)
            for r in rel_list:
                rel_entity_count[r] += 1
                for ri in rel_list:
                    rel_cooc[r, ri] += 1

        rel_pair_pmi = np.zeros((n_rel, n_rel), dtype=np.float32)
        total_entities = float(max(1, n_ent))
        for r in range(n_rel):
            for ri in range(n_rel):
                if rel_cooc[r, ri] < min_pmi_count:
                    continue
                numerator = (rel_cooc[r, ri] + 1.0) * total_entities
                denominator = (rel_entity_count[r] + 1.0) * (rel_entity_count[ri] + 1.0)
                rel_pair_pmi[r, ri] = max(0.0, np.log(numerator / max(denominator, 1.0)))

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

        return {
            'neighbor_entities': torch.from_numpy(neighbor_entities),
            'neighbor_relations': torch.from_numpy(neighbor_relations),
            'neighbor_mask': torch.from_numpy(neighbor_mask),
            'rel_pair_pmi': torch.from_numpy(rel_pair_pmi),
            'entity_log_degree': torch.from_numpy(np.log1p(degree).astype(np.float32)),
            'rel_stats': torch.from_numpy(rel_stats),
        }

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
