from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from torch.nn import functional as F, Parameter
import torch
from torch import nn
import numpy as np
import math


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                target_idxs = these_queries[:, 2].cpu().tolist()
                scores, _ = self.forward(these_queries)
                targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_idxs = torch.as_tensor(
                        filter_out, dtype=torch.long, device=scores.device
                    )
                    scores[i, filter_idxs] = -1e6
                    # The target score was saved above. Mask it explicitly without
                    # mutating the shared filtered-evaluation lookup table.
                    scores[i, target_idxs[i]] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks


class SELayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        y = y.expand_as(x)
        return x * y


class ContextGuidedScaleRouter(nn.Module):
    """
    Produces query-specific weights for multi-scale convolution branches.
    The router is driven by relation semantics and precomputed
    relation-pattern statistics.
    """
    def __init__(
            self, num_filters, emb_dim, stat_dim=0, hidden_dim=None,
            dropout=0.1, temperature=1.0, min_branch_weight=0.0,
            residual=False, residual_init=0.10
    ):
        super(ContextGuidedScaleRouter, self).__init__()
        hidden_dim = hidden_dim or emb_dim
        self.num_filters = num_filters
        self.temperature = temperature
        self.min_branch_weight = min_branch_weight
        self.residual = residual
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        in_dim = emb_dim + stat_dim
        self.router = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_filters),
        )

    def forward(self, rel_emb, rel_stats=None):
        if rel_stats is None:
            route_input = rel_emb
        else:
            route_input = torch.cat([rel_emb, rel_stats], dim=-1)
        logits = self.router(route_input) / max(self.temperature, 1e-6)
        alpha = torch.softmax(logits, dim=-1)
        if self.min_branch_weight > 0:
            min_weight = min(self.min_branch_weight, 1.0 / self.num_filters - 1e-6)
            alpha = (1.0 - min_weight * self.num_filters) * alpha + min_weight
        if self.residual:
            residual_scale = torch.sigmoid(self.residual_logit)
            return 1.0 + residual_scale * (alpha * self.num_filters - 1.0)
        return alpha


class RelationEvidenceMemory(nn.Module):
    """
    Adds query-specific structural evidence to entity logits.

    The module is residual by design: path/type evidence is controlled by
    relation-conditioned gates initialized with small values.
    """
    def __init__(
            self, num_ent, num_rel, emb_dim, stat_dim=0, evidence_context=None,
            use_path=True, use_type=True, gate_hidden=None, gate_dropout=0.05,
            path_strength=0.10, type_strength=0.04,
            path_gate_init=0.05, type_gate_init=0.05,
    ):
        super(RelationEvidenceMemory, self).__init__()
        evidence_context = evidence_context or {}
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.use_path = use_path and all(k in evidence_context for k in [
            'path_query_index', 'path_candidate_ids', 'path_candidate_scores'
        ])
        self.use_type = use_type and 'type_scores' in evidence_context
        self.path_strength = path_strength
        self.type_strength = type_strength

        if self.use_path:
            self.register_buffer('path_query_index', evidence_context['path_query_index'].long(), persistent=False)
            self.register_buffer('path_candidate_ids', evidence_context['path_candidate_ids'].long(), persistent=False)
            self.register_buffer('path_candidate_scores', evidence_context['path_candidate_scores'].float(), persistent=False)
        else:
            self.register_buffer('path_query_index', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_ids', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_scores', torch.empty(0), persistent=False)

        if self.use_type:
            self.register_buffer('type_scores', evidence_context['type_scores'].float(), persistent=False)
        else:
            self.register_buffer('type_scores', torch.empty(num_rel, 0), persistent=False)

        in_dim = emb_dim + stat_dim
        hidden = gate_hidden or emb_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden, 2),
        )
        self._init_gate(path_gate_init, type_gate_init)

    @staticmethod
    def _to_logit(value):
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def _init_gate(self, path_gate_init, type_gate_init):
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        last.bias.data[0] = self._to_logit(path_gate_init)
        last.bias.data[1] = self._to_logit(type_gate_init)

    def forward(self, logits, heads, rels, rel_emb, rel_stats=None):
        if not self.use_path and not self.use_type:
            return logits

        if rel_stats is None:
            gate_input = rel_emb
        else:
            gate_input = torch.cat([rel_emb, rel_stats], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))

        if self.use_type and self.type_scores.numel() > 0:
            type_gate = gates[:, 1] * self.type_strength
            logits = logits + type_gate.unsqueeze(1) * self.type_scores[rels]

        if self.use_path and self.path_candidate_ids.numel() > 0:
            q_idx = self.path_query_index[heads, rels]
            valid_query = (q_idx >= 0).float().unsqueeze(1)
            safe_q_idx = torch.clamp(q_idx, min=0)
            candidate_ids = self.path_candidate_ids[safe_q_idx]
            candidate_scores = self.path_candidate_scores[safe_q_idx] * valid_query
            path_gate = gates[:, 0] * self.path_strength
            path_add = candidate_scores * path_gate.unsqueeze(1)
            logits = logits.scatter_add(1, candidate_ids, path_add)

        return logits


class RelationSpecificConv(torch.nn.Module):
    def __init__(self, num_rel, in_channel, output_channel, filter_size, reshape_H, reshape_W, init_fn, emb_dim=200):
        super(RelationSpecificConv, self).__init__()
        self.num_rel = num_rel
        self.in_channel = in_channel
        self.output_channel = output_channel
        self.h = filter_size[0]
        self.w = filter_size[1]
        self.dilate_height_rate = 1
        self.dilate_width_rate = 1

        if len(filter_size) == 3:
            self.dilate_height_rate = filter_size[2]
            self.dilate_width_rate = filter_size[2]
        if len(filter_size) == 4:
            self.dilate_height_rate = filter_size[2]
            self.dilate_width_rate = filter_size[3]
        filter_dim = self.in_channel * self.output_channel * self.h * self.w
        if emb_dim is not None:
            self.map = torch.nn.Linear(emb_dim, filter_dim)
        else:
            self.filter = torch.nn.Embedding(num_rel, filter_dim, padding_idx=0)
        self.reshape_H, self.reshape_W = reshape_H, reshape_W
        self.init_fn = init_fn
        self.bn = torch.nn.BatchNorm2d(self.output_channel)
        self.se = SELayer(self.output_channel, reduction=int(0.5 * output_channel))

    def init_weights(self):
        if hasattr(self, 'map'):
            self.init_fn(self.map.weight)
        else:
            self.init_fn(self.filter.weight)

    def forward(self, e1_embedded, x, rel, rel_embedded=None):
        if rel_embedded is not None:
            f1 = self.map(rel_embedded)
        else:
            f1 = self.filter(rel)
        f1 = f1.reshape(e1_embedded.size(0) * self.in_channel * self.output_channel, 1, self.h, self.w)
        if self.dilate_height_rate == 1 and self.dilate_width_rate == 1:
            x = F.conv2d(x, f1, groups=e1_embedded.size(0),
                         padding=(int((self.h - 1) // 2), int((self.w - 1) // 2)))
        else:
            x = F.conv2d(x, f1, groups=e1_embedded.size(0),
                         padding=(int((self.h - 1) * self.dilate_height_rate // 2),
                                  int((self.w - 1) * self.dilate_width_rate // 2)),
                         dilation=(self.dilate_height_rate, self.dilate_width_rate))
        x = x.reshape(e1_embedded.size(0), self.output_channel, self.reshape_H, self.reshape_W)
        x = self.bn(x)
        x = self.se(x)
        return x


class MSDCSE(KBCModel):
    def __init__(self, num_ent, num_rel, embedding_dim=300, input_drop=0.4, hidden_drop=0.3, feature_map_drop=0.3,
                 k_w=10, k_h=20, output_channel=20,
                 filter_size_list=[(1, 5), (3, 3), (1, 9)],
                 active_fn='relu', init_fn='xavier_normal', ce_weight=None,
                 use_scale_router=False, relation_context=None,
                 router_hidden=None, router_dropout=0.1, router_temperature=1.0,
                 router_min_branch_weight=0.0, router_residual=False,
                 router_residual_init=0.10,
                 use_rcem=False, rcem_context=None,
                 rcem_use_path=True, rcem_use_type=True,
                 rcem_gate_hidden=None, rcem_gate_dropout=0.05,
                 rcem_path_strength=0.10, rcem_type_strength=0.04,
                 rcem_path_gate_init=0.05, rcem_type_gate_init=0.05):
        super(MSDCSE, self).__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_ent, embedding_dim),
            nn.Embedding(num_rel, embedding_dim),
        ])
        self.emb_ent = self.embeddings[0]
        self.emb_rel = self.embeddings[1]
        self.embedding_dim = embedding_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.perm = 1

        self.k_w = k_w
        self.k_h = k_h
        self.ce_weight = ce_weight
        if self.ce_weight is not None:
            self.loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=ce_weight)
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.active_fn = self.get_active_fn(active_fn)
        self.init_fn = self.get_init_fn(init_fn)
        self.chequer_perm = self.get_chequer_perm()
        print("Chequer perm:", self.chequer_perm)
        self.reshape_H = self.k_w * 2
        self.reshape_W = self.k_h
        self.in_channel = 1
        self.num_filters = len(filter_size_list)
        self.filter_size_list = filter_size_list
        self.share = True
        self.use_scale_router = use_scale_router
        self.use_rcem = use_rcem

        if isinstance(output_channel, int):
            self.output_channels_list = [output_channel] * self.num_filters
        else:
            self.output_channels_list = output_channel

        if len(self.output_channels_list) != self.num_filters:
            raise ValueError("output_channels 长度必须与 filter_sizes 匹配")

        self.conv_layers = torch.nn.ModuleList()
        for i, (out_ch, filter_size) in enumerate(zip(self.output_channels_list, filter_size_list)):
            conv = RelationSpecificConv(
                num_rel=num_rel,
                in_channel=self.in_channel,
                output_channel=out_ch,
                filter_size=filter_size,
                reshape_H=self.reshape_H,
                reshape_W=self.reshape_W,
                init_fn=self.init_fn,
                emb_dim=self.embedding_dim
            )
            self.conv_layers.append(conv)
        total_channel = sum(self.output_channels_list)

        self.input_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channel)
        self.bn1 = torch.nn.BatchNorm2d(total_channel)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        fc_length = self.reshape_H * self.reshape_W * total_channel
        self.fc = torch.nn.Linear(fc_length, embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_ent)))

        self.stat_dim = 0
        if relation_context is not None and 'rel_stats' in relation_context:
            rel_stats = relation_context['rel_stats'].float()
            self.stat_dim = rel_stats.shape[1]
            self.register_buffer('rel_stats', rel_stats, persistent=False)
        else:
            self.register_buffer('rel_stats', torch.empty(num_rel, 0), persistent=False)

        if self.use_scale_router:
            self.scale_router = ContextGuidedScaleRouter(
                num_filters=self.num_filters,
                emb_dim=embedding_dim,
                stat_dim=self.stat_dim,
                hidden_dim=router_hidden,
                dropout=router_dropout,
                temperature=router_temperature,
                min_branch_weight=router_min_branch_weight,
                residual=router_residual,
                residual_init=router_residual_init,
            )
        else:
            self.scale_router = None

        if self.use_rcem:
            self.rcem = RelationEvidenceMemory(
                num_ent=num_ent,
                num_rel=num_rel,
                emb_dim=embedding_dim,
                stat_dim=self.stat_dim,
                evidence_context=rcem_context,
                use_path=rcem_use_path,
                use_type=rcem_use_type,
                gate_hidden=rcem_gate_hidden,
                gate_dropout=rcem_gate_dropout,
                path_strength=rcem_path_strength,
                type_strength=rcem_type_strength,
                path_gate_init=rcem_path_gate_init,
                type_gate_init=rcem_type_gate_init,
            )
        else:
            self.rcem = None

    # ===================== 工具方法 =====================
    def to_var(self, x, use_gpu=True):
        if use_gpu:
            if isinstance(x, torch.Tensor):
                return x.long().to(self.device) if use_gpu else x.long()
            else:
                tensor = torch.as_tensor(x, dtype=torch.long)
                return tensor.to(self.device) if use_gpu else tensor

    def get_active_fn(self, active_fn_name):
        fn_map = {
            'relu': F.relu, 'leaky_relu': F.leaky_relu, 'tanh': F.tanh,
            'sigmoid': F.sigmoid, 'silu': F.silu, 'softplus': F.softplus,
            'gelu': F.gelu, 'elu': F.elu, 'selu': F.selu,
        }
        if active_fn_name not in fn_map:
            raise ValueError("Unsupported activation function: {}".format(active_fn_name))
        return fn_map[active_fn_name]

    def get_init_fn(self, init_fn_name):
        fn_map = {
            'xavier_normal': torch.nn.init.xavier_normal_,
            'xavier_uniform': torch.nn.init.xavier_uniform_,
            'kaiming_normal': torch.nn.init.kaiming_normal_,
            'kaiming_uniform': torch.nn.init.kaiming_uniform_,
        }
        return fn_map.get(init_fn_name, torch.nn.init.xavier_normal_)

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
            comb_idx.append(temp)
        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm

    # ===================== 初始化 =====================
    def init(self):
        init_fn = self.init_fn
        init_fn(self.emb_ent.weight.data)
        init_fn(self.emb_rel.weight.data)
        for conv_layer in self.conv_layers:
            conv_layer.init_weights()

    # ===================== 前向传播 =====================
    def forward(self, x):
        """
        前向传播。
        """
        e1 = x[:, 0]
        rel = x[:, 1]
        e2 = x[:, 2]

        z, e1_embedded, rel_embedded = self._calcate_emebedding(e1, rel)
        e2_embedded = self.emb_ent(e2)

        weight = self.emb_ent.weight.transpose(1, 0)
        pred = torch.mm(z, weight)
        pred = pred + self.b.expand_as(pred)

        if self.rcem is not None:
            rel_stats = self.rel_stats[rel] if self.rel_stats.numel() > 0 else None
            pred = self.rcem(
                pred, e1, rel, rel_embedded, rel_stats=rel_stats
            )

        return pred, [(e1_embedded, rel_embedded, e2_embedded)]

    # ===================== 嵌入计算核心 =====================
    def _calcate_emebedding(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e1_embedded = self.emb_ent(e1)
        rel_embedded = self.emb_rel(rel)
        rel_stats = self.rel_stats[rel] if self.rel_stats.numel() > 0 else None
        comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.perm, 2 * self.k_w, self.k_h))
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
        x = x.permute(1, 0, 2, 3)

        outputs = []
        for conv in self.conv_layers:
            output = conv(e1_embedded, x, rel, rel_embedded)
            outputs.append(output)

        if self.scale_router is not None:
            alpha = self.scale_router(rel_embedded, rel_stats=rel_stats)
            outputs = [
                output * alpha[:, i].view(-1, 1, 1, 1)
                for i, output in enumerate(outputs)
            ]

        x = torch.cat(outputs, dim=1)
        x = self.active_fn(x)
        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.active_fn(x)
        return x, e1_embedded, rel_embedded
