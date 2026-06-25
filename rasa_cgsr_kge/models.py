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
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
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


class RelationAwareAnchorEncoder(nn.Module):
    """
    Relation-aware structural anchor encoder.

    For each query (h, r, ?), the module reads a fixed-size neighborhood table
    for h and lets the current relation select useful neighboring evidence.
    The resulting anchor is fused into the head embedding before convolution.
    """
    def __init__(
            self, num_ent, num_rel, emb_dim, graph_context, stat_dim=0,
            hidden_dim=None, dropout=0.1, pmi_weight=0.15, hub_weight=0.05,
            use_prior=True, residual_init=0.10, gate_bias=-2.0
    ):
        super(RelationAwareAnchorEncoder, self).__init__()
        hidden_dim = hidden_dim or emb_dim
        self.emb_dim = emb_dim
        self.stat_dim = stat_dim
        self.pmi_weight = pmi_weight
        self.hub_weight = hub_weight
        self.use_prior = use_prior
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))

        self.rel_query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.msg_mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        gate_in = emb_dim * 3 + stat_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, emb_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[0].bias, gate_bias)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        if use_prior:
            self.rel_prior = nn.Embedding(num_rel, emb_dim)
        else:
            self.rel_prior = None

        self.register_buffer(
            'neighbor_entities',
            graph_context['neighbor_entities'].long(),
            persistent=False
        )
        self.register_buffer(
            'neighbor_relations',
            graph_context['neighbor_relations'].long(),
            persistent=False
        )
        self.register_buffer(
            'neighbor_mask',
            graph_context['neighbor_mask'].float(),
            persistent=False
        )
        self.register_buffer(
            'rel_pair_pmi',
            graph_context['rel_pair_pmi'].float(),
            persistent=False
        )
        self.register_buffer(
            'entity_log_degree',
            graph_context['entity_log_degree'].float(),
            persistent=False
        )

    def forward(self, ent_ids, rel_ids, ent_emb, rel_emb, ent_embedding, rel_embedding, rel_stats=None):
        neigh_ent = self.neighbor_entities[ent_ids]
        neigh_rel = self.neighbor_relations[ent_ids]
        mask = self.neighbor_mask[ent_ids]

        neigh_ent_emb = ent_embedding(neigh_ent)
        neigh_rel_emb = rel_embedding(neigh_rel)

        rel_query = self.rel_query(rel_emb).unsqueeze(1)
        scores = torch.sum(rel_query * neigh_rel_emb, dim=-1) / math.sqrt(self.emb_dim)

        if self.pmi_weight != 0:
            scores = scores + self.pmi_weight * self.rel_pair_pmi[rel_ids.unsqueeze(1), neigh_rel]
        if self.hub_weight != 0:
            scores = scores - self.hub_weight * self.entity_log_degree[neigh_ent]

        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=1) * mask
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-9)

        msg = self.msg_mlp(torch.cat([neigh_ent_emb, neigh_rel_emb, neigh_ent_emb * neigh_rel_emb], dim=-1))
        anchor = torch.sum(attn.unsqueeze(-1) * msg, dim=1)

        has_neighbor = (mask.sum(dim=1, keepdim=True) > 0).float()
        if self.rel_prior is not None:
            prior = self.rel_prior(rel_ids)
            anchor = has_neighbor * anchor + (1.0 - has_neighbor) * prior
        anchor = self.dropout(self.norm(anchor))

        if rel_stats is None:
            gate_input = torch.cat([ent_emb, rel_emb, anchor], dim=-1)
        else:
            gate_input = torch.cat([ent_emb, rel_emb, anchor, rel_stats], dim=-1)
        gate = self.gate(gate_input)
        residual_scale = torch.sigmoid(self.residual_logit)
        enhanced_ent = ent_emb + residual_scale * gate * anchor
        return enhanced_ent, anchor, gate


class ContextGuidedScaleRouter(nn.Module):
    """
    Produces query-specific weights for multi-scale convolution branches.
    The router is driven by relation semantics, structural anchor, and
    precomputed relation-pattern statistics.
    """
    def __init__(
            self, num_filters, emb_dim, stat_dim=0, hidden_dim=None,
            dropout=0.1, temperature=1.0, min_branch_weight=0.0,
            residual=True, residual_init=0.10
    ):
        super(ContextGuidedScaleRouter, self).__init__()
        hidden_dim = hidden_dim or emb_dim
        self.num_filters = num_filters
        self.temperature = temperature
        self.min_branch_weight = min_branch_weight
        self.residual = residual
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        in_dim = emb_dim * 3 + stat_dim
        self.router = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_filters),
        )

    def forward(self, rel_emb, anchor, rel_stats=None):
        if rel_stats is None:
            route_input = torch.cat([rel_emb, anchor, rel_emb * anchor], dim=-1)
        else:
            route_input = torch.cat([rel_emb, anchor, rel_emb * anchor, rel_stats], dim=-1)
        logits = self.router(route_input) / max(self.temperature, 1e-6)
        alpha = torch.softmax(logits, dim=-1)
        if self.min_branch_weight > 0:
            min_weight = min(self.min_branch_weight, 1.0 / self.num_filters - 1e-6)
            alpha = (1.0 - min_weight * self.num_filters) * alpha + min_weight
        if self.residual:
            residual_scale = torch.sigmoid(self.residual_logit)
            return 1.0 + residual_scale * (alpha * self.num_filters - 1.0)
        return alpha


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
                 use_anchor=False, use_scale_router=False, graph_context=None,
                 anchor_hidden=None, anchor_dropout=0.1, anchor_pmi_weight=0.15,
                 anchor_hub_weight=0.05, anchor_use_prior=True,
                 anchor_residual_init=0.10, anchor_gate_bias=-2.0,
                 router_hidden=None, router_dropout=0.1, router_temperature=1.0,
                 router_min_branch_weight=0.0, router_residual=True,
                 router_residual_init=0.10):
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
        self.use_anchor = use_anchor
        self.use_scale_router = use_scale_router

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
        if graph_context is not None and 'rel_stats' in graph_context:
            rel_stats = graph_context['rel_stats'].float()
            self.stat_dim = rel_stats.shape[1]
            self.register_buffer('rel_stats', rel_stats, persistent=False)
        else:
            self.register_buffer('rel_stats', torch.empty(num_rel, 0), persistent=False)

        if self.use_anchor:
            if graph_context is None:
                raise ValueError("use_anchor=True requires graph_context from Dataset.build_graph_context().")
            self.anchor_encoder = RelationAwareAnchorEncoder(
                num_ent=num_ent,
                num_rel=num_rel,
                emb_dim=embedding_dim,
                graph_context=graph_context,
                stat_dim=self.stat_dim,
                hidden_dim=anchor_hidden,
                dropout=anchor_dropout,
                pmi_weight=anchor_pmi_weight,
                hub_weight=anchor_hub_weight,
                use_prior=anchor_use_prior,
                residual_init=anchor_residual_init,
                gate_bias=anchor_gate_bias,
            )
        else:
            self.anchor_encoder = None

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

        return pred, [(e1_embedded, rel_embedded, e2_embedded)]

    # ===================== 嵌入计算核心 =====================
    def _calcate_emebedding(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e1_embedded = self.emb_ent(e1)
        rel_embedded = self.emb_rel(rel)
        rel_stats = self.rel_stats[rel] if self.rel_stats.numel() > 0 else None
        anchor = torch.zeros_like(e1_embedded)

        if self.anchor_encoder is not None:
            e1_contextual, anchor, _ = self.anchor_encoder(
                e1, rel, e1_embedded, rel_embedded,
                self.emb_ent, self.emb_rel, rel_stats=rel_stats
            )
        else:
            e1_contextual = e1_embedded

        comb_emb = torch.cat([e1_contextual, rel_embedded], dim=1)
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
            alpha = self.scale_router(rel_embedded, anchor, rel_stats=rel_stats)
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
