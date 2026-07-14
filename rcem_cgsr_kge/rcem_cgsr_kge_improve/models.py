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
                    # Work on a copy: evaluation must not mutate the persistent
                    # filtered-ranking dictionary across epochs.
                    filter_out = list(filters[(query[0].item(), query[1].item())])
                    filter_out.append(queries[b_begin + i, 2].item())
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


class RelationContextEncoder(nn.Module):
    """
    Encodes a relation embedding together with leakage-free relation statistics.

    The encoded context is shared by the scale router and RCEM.  This makes the
    two improvements part of one relation-heterogeneity-aware mechanism instead
    of two independent feature additions.
    """
    def __init__(self, emb_dim, stat_dim, context_dim=None, hidden_dim=None, dropout=0.1):
        super(RelationContextEncoder, self).__init__()
        self.context_dim = context_dim or emb_dim
        hidden_dim = hidden_dim or self.context_dim
        in_dim = emb_dim + stat_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.context_dim),
            nn.LayerNorm(self.context_dim),
        )

    def forward(self, rel_emb, rel_stats=None):
        if rel_stats is None:
            context_input = rel_emb
        else:
            context_input = torch.cat([rel_emb, rel_stats], dim=-1)
        return self.encoder(context_input)


class ContextGuidedScaleRouter(nn.Module):
    """
    Produces query-specific weights for multi-scale convolution branches.
    The router is driven by relation semantics and precomputed
    relation-pattern statistics.
    """
    def __init__(
            self, num_filters, emb_dim, stat_dim=0, hidden_dim=None,
            dropout=0.1, temperature=1.0, min_branch_weight=0.0,
            residual=False, residual_init=0.10,
            query_dim=0, context_dim=0, use_query_context=False,
            energy_preserving=True
    ):
        super(ContextGuidedScaleRouter, self).__init__()
        hidden_dim = hidden_dim or emb_dim
        self.num_filters = num_filters
        self.temperature = temperature
        self.min_branch_weight = min_branch_weight
        self.residual = residual
        self.query_dim = query_dim if use_query_context else 0
        self.context_dim = context_dim
        self.use_query_context = use_query_context
        self.energy_preserving = energy_preserving
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        if context_dim > 0:
            in_dim = emb_dim + context_dim + self.query_dim
        else:
            in_dim = emb_dim + stat_dim + self.query_dim
        self.router = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_filters),
        )

    def forward(self, rel_emb, rel_stats=None, query_emb=None, context_emb=None):
        if context_emb is not None:
            route_parts = [rel_emb, context_emb]
        else:
            route_parts = [rel_emb]
            if rel_stats is not None:
                route_parts.append(rel_stats)
        if self.use_query_context:
            if query_emb is None:
                query_emb = torch.zeros(
                    rel_emb.size(0), self.query_dim,
                    device=rel_emb.device, dtype=rel_emb.dtype
                )
            route_parts.append(query_emb)
        route_input = torch.cat(route_parts, dim=-1)
        logits = self.router(route_input) / max(self.temperature, 1e-6)
        probabilities = torch.softmax(logits, dim=-1)
        if self.min_branch_weight > 0:
            min_weight = min(self.min_branch_weight, 1.0 / self.num_filters - 1e-6)
            probabilities = (1.0 - min_weight * self.num_filters) * probabilities + min_weight
        alpha = probabilities * self.num_filters if self.energy_preserving else probabilities
        if self.residual:
            residual_scale = torch.sigmoid(self.residual_logit)
            if not self.energy_preserving:
                alpha = alpha * self.num_filters
            return 1.0 + residual_scale * (alpha - 1.0)
        return alpha


class RelationEvidenceMemory(nn.Module):
    """
    Adds query-specific structural evidence to entity logits.

    The module is residual by design: evidence is controlled by a competitive
    base/path/role gate and by query-level evidence quality features.  Path
    candidates are mined offline for efficiency, while a small calibrator learns
    to correct their reliability from the KGE objective.
    """
    def __init__(
            self, num_ent, num_rel, emb_dim, stat_dim=0, evidence_context=None,
            use_path=True, use_type=True, gate_hidden=None, gate_dropout=0.05,
            path_strength=0.10, type_strength=0.04,
            path_gate_init=0.05, type_gate_init=0.05,
            query_dim=0, context_dim=0, use_query_gate=True,
            use_candidate_calibrator=True, calibrator_hidden=None,
            calibrator_strength=0.25,
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
        self.query_dim = query_dim if use_query_gate else 0
        self.context_dim = context_dim
        self.use_query_gate = use_query_gate
        self.calibrator_strength = calibrator_strength

        if self.use_path:
            self.register_buffer('path_query_index', evidence_context['path_query_index'].long(), persistent=False)
            self.register_buffer('path_candidate_ids', evidence_context['path_candidate_ids'].long(), persistent=False)
            self.register_buffer('path_candidate_scores', evidence_context['path_candidate_scores'].float(), persistent=False)
        else:
            self.register_buffer('path_query_index', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_ids', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('path_candidate_scores', torch.empty(0), persistent=False)

        if self.use_type:
            type_scores = evidence_context['type_scores'].float()
            self.register_buffer('type_scores', type_scores, persistent=False)
            type_mean = type_scores.mean(dim=1)
            type_std = type_scores.std(dim=1, unbiased=False)
            type_max = type_scores.max(dim=1).values
            type_prob = type_scores / type_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
            type_entropy = -torch.sum(type_prob * torch.log(type_prob.clamp_min(1e-8)), dim=1)
            type_entropy = type_entropy / math.log(max(type_scores.shape[1], 2))
            self.register_buffer(
                'type_quality',
                torch.stack([type_mean, type_std, type_max, type_entropy], dim=1),
                persistent=False,
            )
        else:
            self.register_buffer('type_scores', torch.empty(num_rel, 0), persistent=False)
            self.register_buffer('type_quality', torch.empty(num_rel, 4), persistent=False)

        # Five path-quality features: validity, candidate density, mean score,
        # maximum score, and normalized entropy.  Four type-quality features are
        # precomputed per relation above.
        self.quality_dim = 9
        if context_dim > 0:
            in_dim = emb_dim + context_dim + self.query_dim + self.quality_dim
        else:
            in_dim = emb_dim + stat_dim + self.query_dim + self.quality_dim
        hidden = gate_hidden or emb_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden, 3),
        )
        self.register_buffer(
            'gate_active_mask',
            torch.tensor([True, self.use_path, self.use_type], dtype=torch.bool),
            persistent=False,
        )

        self.path_calibrator = None
        if self.use_path and use_candidate_calibrator:
            calibrator_hidden = calibrator_hidden or max(16, min(128, emb_dim // 4))
            self.path_calibrator = nn.Sequential(
                nn.Linear(8, calibrator_hidden),
                nn.LayerNorm(calibrator_hidden),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(calibrator_hidden, 1),
            )
            # Start from the deterministic evidence table and let training learn
            # only a bounded correction.
            nn.init.zeros_(self.path_calibrator[-1].weight)
            nn.init.zeros_(self.path_calibrator[-1].bias)

        self._init_gate(path_gate_init, type_gate_init)

    @staticmethod
    def _to_logit(value):
        value = min(max(float(value), 1e-5), 1.0 - 1e-5)
        return math.log(value / (1.0 - value))

    def _init_gate(self, path_gate_init, type_gate_init):
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        base_init = max(1e-4, 1.0 - float(path_gate_init) - float(type_gate_init))
        normalizer = base_init + float(path_gate_init) + float(type_gate_init)
        probabilities = torch.tensor([
            base_init / normalizer,
            float(path_gate_init) / normalizer,
            float(type_gate_init) / normalizer,
        ], dtype=last.bias.dtype, device=last.bias.device)
        last.bias.data.copy_(torch.log(probabilities.clamp_min(1e-5)))

    @staticmethod
    def _path_quality(candidate_scores, valid_query):
        valid = (candidate_scores > 0).float() * valid_query
        count = valid.sum(dim=1)
        max_candidates = max(candidate_scores.shape[1], 1)
        score_sum = candidate_scores.sum(dim=1)
        mean_score = score_sum / count.clamp_min(1.0)
        max_score = candidate_scores.max(dim=1).values
        probabilities = candidate_scores / score_sum.unsqueeze(1).clamp_min(1e-6)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=1
        ) / math.log(max(max_candidates, 2))
        return torch.stack([
            valid_query.squeeze(1),
            count / float(max_candidates),
            mean_score,
            max_score,
            entropy,
        ], dim=1)

    def _gate_input(self, rel_emb, rel_stats, head_emb, context_emb, quality):
        parts = [rel_emb]
        if context_emb is not None:
            parts.append(context_emb)
        elif rel_stats is not None:
            parts.append(rel_stats)
        if self.use_query_gate:
            if head_emb is None:
                head_emb = torch.zeros(
                    rel_emb.size(0), self.query_dim,
                    device=rel_emb.device, dtype=rel_emb.dtype
                )
            parts.append(head_emb)
        parts.append(quality)
        return torch.cat(parts, dim=-1)

    def forward(
            self, logits, heads, rels, rel_emb, rel_stats=None,
            head_emb=None, context_emb=None, module_scale=1.0
    ):
        if (not self.use_path and not self.use_type) or module_scale <= 0.0:
            return logits

        batch_size = logits.size(0)
        device = logits.device
        zero_path_quality = torch.zeros(batch_size, 5, device=device, dtype=logits.dtype)

        candidate_ids = None
        candidate_scores = None
        if self.use_path and self.path_candidate_ids.numel() > 0:
            q_idx = self.path_query_index[heads, rels]
            valid_query = (q_idx >= 0).float().unsqueeze(1)
            safe_q_idx = torch.clamp(q_idx, min=0)
            candidate_ids = self.path_candidate_ids[safe_q_idx]
            candidate_scores = self.path_candidate_scores[safe_q_idx] * valid_query
            path_quality = self._path_quality(candidate_scores, valid_query)
        else:
            valid_query = torch.zeros(batch_size, 1, device=device, dtype=logits.dtype)
            path_quality = zero_path_quality

        if self.use_type and self.type_scores.numel() > 0:
            type_quality = self.type_quality[rels]
        else:
            type_quality = torch.zeros(batch_size, 4, device=device, dtype=logits.dtype)

        quality = torch.cat([path_quality, type_quality], dim=1)
        gate_input = self._gate_input(rel_emb, rel_stats, head_emb, context_emb, quality)
        gate_logits = self.gate(gate_input)
        gate_logits = gate_logits.masked_fill(
            ~self.gate_active_mask.view(1, 3), -1e4
        )
        gate_probabilities = torch.softmax(gate_logits, dim=-1)

        if self.use_type and self.type_scores.numel() > 0:
            type_evidence = self.type_scores[rels]
            type_mean = type_evidence.mean(dim=1, keepdim=True)
            type_std = type_evidence.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
            type_evidence = torch.tanh((type_evidence - type_mean) / type_std)
            type_gate = gate_probabilities[:, 2] * self.type_strength * module_scale
            logits = logits + type_gate.unsqueeze(1) * type_evidence

        if candidate_ids is not None and candidate_scores is not None:
            calibrated_scores = candidate_scores
            if self.path_calibrator is not None:
                base_mean = logits.mean(dim=1, keepdim=True)
                base_std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
                candidate_base = logits.gather(1, candidate_ids)
                candidate_base = (candidate_base - base_mean) / base_std

                if self.use_type and self.type_scores.numel() > 0:
                    candidate_type = self.type_scores[rels].gather(1, candidate_ids)
                else:
                    candidate_type = torch.zeros_like(candidate_scores)

                expanded_quality = path_quality.unsqueeze(1).expand(
                    -1, candidate_scores.size(1), -1
                )
                calibrator_features = torch.cat([
                    candidate_base.unsqueeze(-1),
                    candidate_scores.unsqueeze(-1),
                    candidate_type.unsqueeze(-1),
                    expanded_quality,
                ], dim=-1)
                correction = self.path_calibrator(calibrator_features).squeeze(-1)
                calibrated_scores = candidate_scores + self.calibrator_strength * torch.tanh(correction)
                calibrated_scores = calibrated_scores * (candidate_scores > 0).float()

            path_gate = gate_probabilities[:, 1] * self.path_strength * module_scale
            path_add = calibrated_scores * path_gate.unsqueeze(1)
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
                 module_warmup_epochs=0, module_ramp_epochs=1,
                 router_hidden=None, router_dropout=0.1, router_temperature=1.0,
                 router_min_branch_weight=0.0, router_residual=False,
                 router_residual_init=0.10, router_use_query_context=True,
                 router_energy_preserving=True, relation_context_dim=None,
                 relation_context_hidden=None, relation_context_dropout=0.1,
                 use_rcem=False, rcem_context=None,
                 rcem_use_path=True, rcem_use_type=True,
                 rcem_warmup_epochs=0, rcem_ramp_epochs=1,
                 rcem_gate_hidden=None, rcem_gate_dropout=0.05,
                 rcem_path_strength=0.10, rcem_type_strength=0.04,
                 rcem_path_gate_init=0.05, rcem_type_gate_init=0.05,
                 rcem_use_query_gate=True, rcem_use_candidate_calibrator=True,
                 rcem_calibrator_hidden=None, rcem_calibrator_strength=0.25):
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
        self.module_warmup_epochs = module_warmup_epochs
        self.module_ramp_epochs = max(1, module_ramp_epochs)
        self.current_epoch = 0
        self.use_rcem = use_rcem
        self.rcem_warmup_epochs = rcem_warmup_epochs
        self.rcem_ramp_epochs = max(1, rcem_ramp_epochs)

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

        self.relation_context_dim = relation_context_dim or embedding_dim
        self.relation_context_encoder = None
        if self.stat_dim > 0 and (use_scale_router or use_rcem):
            self.relation_context_encoder = RelationContextEncoder(
                emb_dim=embedding_dim,
                stat_dim=self.stat_dim,
                context_dim=self.relation_context_dim,
                hidden_dim=relation_context_hidden,
                dropout=relation_context_dropout,
            )

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
                query_dim=embedding_dim,
                context_dim=self.relation_context_dim if self.relation_context_encoder is not None else 0,
                use_query_context=router_use_query_context,
                energy_preserving=router_energy_preserving,
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
                query_dim=embedding_dim,
                context_dim=self.relation_context_dim if self.relation_context_encoder is not None else 0,
                use_query_gate=rcem_use_query_gate,
                use_candidate_calibrator=rcem_use_candidate_calibrator,
                calibrator_hidden=rcem_calibrator_hidden,
                calibrator_strength=rcem_calibrator_strength,
            )
        else:
            self.rcem = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_module_scale(self):
        if not self.training:
            return 1.0
        if self.current_epoch < self.module_warmup_epochs:
            return 0.0
        progress = (self.current_epoch - self.module_warmup_epochs + 1) / float(self.module_ramp_epochs)
        return min(1.0, max(0.0, progress))

    def get_rcem_scale(self):
        if not self.training:
            return 1.0
        if self.current_epoch < self.rcem_warmup_epochs:
            return 0.0
        progress = (self.current_epoch - self.rcem_warmup_epochs + 1) / float(self.rcem_ramp_epochs)
        return min(1.0, max(0.0, progress))

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

        z, e1_embedded, rel_embedded, relation_context_embedded = self._calcate_emebedding(e1, rel)
        e2_embedded = self.emb_ent(e2)

        weight = self.emb_ent.weight.transpose(1, 0)
        pred = torch.mm(z, weight)
        pred = pred + self.b.expand_as(pred)

        if self.rcem is not None:
            rel_stats = self.rel_stats[rel] if self.rel_stats.numel() > 0 else None
            pred = self.rcem(
                pred, e1, rel, rel_embedded, rel_stats=rel_stats,
                head_emb=e1_embedded, context_emb=relation_context_embedded,
                module_scale=self.get_rcem_scale()
            )

        return pred, [(e1_embedded, rel_embedded, e2_embedded)]

    # ===================== 嵌入计算核心 =====================
    def _calcate_emebedding(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e1_embedded = self.emb_ent(e1)
        rel_embedded = self.emb_rel(rel)
        rel_stats = self.rel_stats[rel] if self.rel_stats.numel() > 0 else None
        if self.relation_context_encoder is not None:
            relation_context_embedded = self.relation_context_encoder(rel_embedded, rel_stats)
        else:
            relation_context_embedded = None
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
            alpha = self.scale_router(
                rel_embedded,
                rel_stats=rel_stats,
                query_emb=e1_embedded,
                context_emb=relation_context_embedded,
            )
            module_scale = self.get_module_scale()
            if isinstance(module_scale, float) and module_scale != 1.0:
                alpha = 1.0 + module_scale * (alpha - 1.0)
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
        return x, e1_embedded, rel_embedded, relation_context_embedded
