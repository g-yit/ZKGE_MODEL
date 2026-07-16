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
    Efficient hierarchical query-content routing for multi-scale branches.

    Relation semantics provide a scale prior, the head-relation interaction and
    train-only entity statistics adapt it to each query, and pooled branch
    responses provide content feedback. Returned gains preserve an average value
    of one, avoiding the feature-scale shrinkage of raw softmax weights.
    """
    def __init__(
            self, num_filters, emb_dim, stat_dim=0, entity_stat_dim=0,
            branch_channels=None, hidden_dim=None,
            dropout=0.1, temperature=1.0, min_branch_weight=0.02,
            residual=True, residual_init=0.10, content_scale=0.25
    ):
        super(ContextGuidedScaleRouter, self).__init__()
        hidden_dim = hidden_dim or min(emb_dim, 128)
        branch_channels = branch_channels or [1] * num_filters
        if len(branch_channels) != num_filters:
            raise ValueError("branch_channels must match num_filters")

        self.num_filters = num_filters
        self.temperature = temperature
        self.min_branch_weight = min_branch_weight
        self.residual = residual
        self.stat_dim = stat_dim
        self.entity_stat_dim = entity_stat_dim
        self.hidden_dim = hidden_dim

        relation_input_dim = emb_dim + stat_dim
        self.relation_projection = nn.Linear(relation_input_dim, hidden_dim)
        self.head_projection = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.interaction_projection = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.entity_stat_projection = (
            nn.Linear(entity_stat_dim, hidden_dim, bias=False)
            if entity_stat_dim > 0 else None
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.relation_prior = nn.Linear(hidden_dim, num_filters)
        self.query_router = nn.Linear(hidden_dim, num_filters)
        # A zero initial scale is a strict content-feedback ablation. Positive
        # values initialize a learnable positive coefficient, preserving the
        # intended hyperparameter experiment without silently re-enabling the
        # branch-content pathway during training.
        self.use_content_feedback = float(content_scale) > 0.0
        if self.use_content_feedback:
            self.content_projections = nn.ModuleList([
                nn.Linear(channel, hidden_dim, bias=False)
                for channel in branch_channels
            ])
            content_scale = float(content_scale)
            content_scale_logit = (
                math.log(math.expm1(content_scale))
                if content_scale < 20.0 else content_scale
            )
            self.content_scale_logit = nn.Parameter(torch.tensor(
                content_scale_logit
            ))
        else:
            self.content_projections = nn.ModuleList()
            self.register_parameter('content_scale_logit', None)

        self.confidence = nn.Linear(hidden_dim, 1)
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(
            self.confidence.bias,
            math.log(residual_init / (1.0 - residual_init))
        )

    def forward(
            self, head_emb, rel_emb, rel_stats=None, entity_stats=None,
            branch_outputs=None
    ):
        if self.stat_dim > 0:
            if rel_stats is None:
                raise ValueError("rel_stats are required by the configured router")
            relation_input = torch.cat([rel_emb, rel_stats], dim=-1)
        else:
            relation_input = rel_emb

        relation_hidden = F.silu(self.relation_projection(relation_input))
        query_hidden = (
            relation_hidden
            + self.head_projection(head_emb)
            + self.interaction_projection(head_emb * rel_emb)
        )
        if self.entity_stat_projection is not None:
            if entity_stats is None:
                raise ValueError("entity_stats are required by the configured router")
            query_hidden = query_hidden + self.entity_stat_projection(entity_stats)

        query_hidden = self.query_norm(query_hidden)
        query_hidden = self.dropout(F.silu(query_hidden))
        prior_logits = self.relation_prior(self.dropout(relation_hidden))
        logits = prior_logits + self.query_router(query_hidden)

        if self.use_content_feedback and branch_outputs is not None:
            if len(branch_outputs) != self.num_filters:
                raise ValueError("branch_outputs must match num_filters")
            content_scores = []
            normalizer = math.sqrt(float(self.hidden_dim))
            for projection, output in zip(self.content_projections, branch_outputs):
                pooled = output.mean(dim=(-2, -1))
                content_token = torch.tanh(projection(pooled))
                content_scores.append(
                    torch.sum(query_hidden * content_token, dim=-1) / normalizer
                )
            content_scores = torch.stack(content_scores, dim=-1)
            logits = logits + F.softplus(self.content_scale_logit) * content_scores

        logits = logits / max(self.temperature, 1e-6)
        alpha = torch.softmax(logits, dim=-1)
        if self.min_branch_weight > 0:
            min_weight = min(self.min_branch_weight, 1.0 / self.num_filters - 1e-6)
            alpha = (1.0 - min_weight * self.num_filters) * alpha + min_weight

        full_gains = alpha * self.num_filters
        if self.residual:
            confidence = torch.sigmoid(self.confidence(query_hidden))
            return 1.0 + confidence * (full_gains - 1.0)
        return full_gains


class RelationEvidenceMemory(nn.Module):
    """
    Adds query-specific structural evidence to entity logits.

    Path/type evidence is controlled by gates conditioned on the encoded query,
    head-relation interaction, relation statistics, and evidence reliability.
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
            path_query_features = evidence_context.get(
                'path_query_features', torch.empty(0, 0)
            ).float()
            self.path_feature_dim = (
                path_query_features.shape[1]
                if path_query_features.ndim == 2 else 0
            )
            self.register_buffer(
                'path_query_features', path_query_features, persistent=False
            )
        else:
            self.register_buffer('path_query_index', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_ids', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_scores', torch.empty(0), persistent=False)
            self.path_feature_dim = 0
            self.register_buffer(
                'path_query_features', torch.empty(0, 0), persistent=False
            )

        if self.use_type:
            self.register_buffer('type_scores', evidence_context['type_scores'].float(), persistent=False)
            type_reliability = evidence_context.get(
                'type_reliability', torch.empty(0, 0)
            ).float()
            self.type_feature_dim = (
                type_reliability.shape[1]
                if type_reliability.ndim == 2 else 0
            )
            self.register_buffer(
                'type_reliability', type_reliability, persistent=False
            )
        else:
            self.register_buffer('type_scores', torch.empty(num_rel, 0), persistent=False)
            self.type_feature_dim = 0
            self.register_buffer(
                'type_reliability', torch.empty(0, 0), persistent=False
            )

        hidden = gate_hidden or min(emb_dim, 128)
        self.stat_dim = stat_dim
        self.gate_relation = nn.Linear(emb_dim + stat_dim, hidden)
        self.gate_head = nn.Linear(emb_dim, hidden, bias=False)
        self.gate_query = nn.Linear(emb_dim, hidden, bias=False)
        self.gate_interaction = nn.Linear(emb_dim, hidden, bias=False)
        self.gate_path_features = (
            nn.Linear(self.path_feature_dim, hidden, bias=False)
            if self.path_feature_dim > 0 else None
        )
        self.gate_type_features = (
            nn.Linear(self.type_feature_dim, hidden, bias=False)
            if self.type_feature_dim > 0 else None
        )
        self.gate_norm = nn.LayerNorm(hidden)
        self.gate_dropout = nn.Dropout(gate_dropout)
        self.gate_output = nn.Linear(hidden, 2)
        self._init_gate(path_gate_init, type_gate_init)

    @staticmethod
    def _to_logit(value):
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def _init_gate(self, path_gate_init, type_gate_init):
        last = self.gate_output
        nn.init.zeros_(last.weight)
        last.bias.data[0] = self._to_logit(path_gate_init)
        last.bias.data[1] = self._to_logit(type_gate_init)

    def forward(
            self, logits, heads, rels, rel_emb, rel_stats=None,
            head_emb=None, query_emb=None
    ):
        if not self.use_path and not self.use_type:
            return logits

        q_idx = None
        valid_query = None
        safe_q_idx = None
        path_features = None
        if self.use_path and self.path_candidate_ids.numel() > 0:
            q_idx = self.path_query_index[heads, rels]
            valid_query = (q_idx >= 0).float().unsqueeze(1)
            safe_q_idx = torch.clamp(q_idx, min=0)
            if self.path_query_features.numel() > 0:
                path_features = (
                    self.path_query_features[safe_q_idx] * valid_query
                )

        type_features = None
        if self.use_type and self.type_reliability.numel() > 0:
            type_features = self.type_reliability[rels]

        if self.stat_dim > 0:
            if rel_stats is None:
                raise ValueError("rel_stats are required by the configured RCEM gate")
            relation_input = torch.cat([rel_emb, rel_stats], dim=-1)
        else:
            relation_input = rel_emb

        gate_hidden = self.gate_relation(relation_input)
        if head_emb is not None:
            gate_hidden = (
                gate_hidden
                + self.gate_head(head_emb)
                + self.gate_interaction(head_emb * rel_emb)
            )
        if query_emb is not None:
            gate_hidden = gate_hidden + self.gate_query(query_emb)
        if self.gate_path_features is not None and path_features is not None:
            gate_hidden = gate_hidden + self.gate_path_features(path_features)
        if self.gate_type_features is not None and type_features is not None:
            gate_hidden = gate_hidden + self.gate_type_features(type_features)

        gate_hidden = self.gate_norm(gate_hidden)
        gate_hidden = self.gate_dropout(F.silu(gate_hidden))
        gates = torch.sigmoid(self.gate_output(gate_hidden))

        if self.use_type and self.type_scores.numel() > 0:
            type_gate = gates[:, 1] * self.type_strength
            logits = logits + type_gate.unsqueeze(1) * self.type_scores[rels]

        if self.use_path and self.path_candidate_ids.numel() > 0:
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
                 context_hidden=None,
                 use_scale_router=False, relation_context=None,
                 router_temperature=1.0, router_residual=True,
                 router_content_scale=0.25,
                 use_rcem=False, rcem_context=None,
                 rcem_use_path=True, rcem_use_type=True,
                 rcem_path_strength=0.10, rcem_type_strength=0.04):
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

        self.entity_stat_dim = 0
        if (
                self.use_scale_router
                and relation_context is not None
                and 'entity_stats' in relation_context
        ):
            entity_stats = relation_context['entity_stats'].float()
            self.entity_stat_dim = entity_stats.shape[1]
            self.register_buffer('entity_stats', entity_stats, persistent=False)
        else:
            self.register_buffer(
                'entity_stats', torch.empty(num_ent, 0), persistent=False
            )

        if self.use_scale_router:
            self.scale_router = ContextGuidedScaleRouter(
                num_filters=self.num_filters,
                emb_dim=embedding_dim,
                stat_dim=self.stat_dim,
                entity_stat_dim=self.entity_stat_dim,
                branch_channels=self.output_channels_list,
                hidden_dim=context_hidden,
                temperature=router_temperature,
                residual=router_residual,
                content_scale=router_content_scale,
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
                gate_hidden=context_hidden,
                path_strength=rcem_path_strength,
                type_strength=rcem_type_strength,
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
                pred, e1, rel, rel_embedded, rel_stats=rel_stats,
                head_emb=e1_embedded, query_emb=z
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
            entity_stats = (
                self.entity_stats[e1] if self.entity_stats.numel() > 0 else None
            )
            branch_gains = self.scale_router(
                head_emb=e1_embedded,
                rel_emb=rel_embedded,
                rel_stats=rel_stats,
                entity_stats=entity_stats,
                branch_outputs=outputs,
            )
            outputs = [
                output * branch_gains[:, i].view(-1, 1, 1, 1)
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
