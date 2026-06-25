import math
from abc import ABC
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


class KBCModel(nn.Module, ABC):
    def get_ranking(
        self,
        queries: torch.Tensor,
        filters: Dict[Tuple[int, int], List[int]],
        batch_size: int = 1000,
        chunk_size: int = -1,
    ):
        ranks = torch.ones(len(queries), device="cpu")
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin : b_begin + batch_size]
                target_idxs = these_queries[:, 2].detach().cpu().tolist()
                scores, _ = self.forward(these_queries)
                targets = torch.stack(
                    [scores[row, col] for row, col in enumerate(target_idxs)]
                ).unsqueeze(-1)

                for i, query in enumerate(these_queries):
                    key = (query[0].item(), query[1].item())
                    filter_out = list(filters.get(key, []))
                    filter_out.append(queries[b_begin + i, 2].item())
                    filter_tensor = torch.as_tensor(
                        filter_out, dtype=torch.long, device=scores.device
                    )
                    scores[i, filter_tensor] = -1e6

                ranks[b_begin : b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).detach().cpu()
                b_begin += batch_size
        return ranks


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        hidden = max(1, channel // max(1, reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RelationSpecificConv(nn.Module):
    def __init__(
        self,
        num_rel,
        in_channel,
        output_channel,
        filter_size,
        reshape_H,
        reshape_W,
        init_fn,
        emb_dim=200,
    ):
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
            self.map = nn.Linear(emb_dim, filter_dim)
        else:
            self.filter = nn.Embedding(num_rel, filter_dim, padding_idx=0)

        self.reshape_H, self.reshape_W = reshape_H, reshape_W
        self.init_fn = init_fn
        self.bn = nn.BatchNorm2d(self.output_channel)
        self.se = SELayer(self.output_channel, reduction=max(1, int(0.5 * output_channel)))

    def init_weights(self):
        if hasattr(self, "map"):
            self.init_fn(self.map.weight)
            if self.map.bias is not None:
                nn.init.zeros_(self.map.bias)
        else:
            self.init_fn(self.filter.weight)

    def forward(self, e1_embedded, x, rel, rel_embedded=None):
        if rel_embedded is not None:
            f1 = self.map(rel_embedded)
        else:
            f1 = self.filter(rel)

        batch_size = e1_embedded.size(0)
        f1 = f1.reshape(batch_size * self.in_channel * self.output_channel, 1, self.h, self.w)
        if self.dilate_height_rate == 1 and self.dilate_width_rate == 1:
            x = F.conv2d(
                x,
                f1,
                groups=batch_size,
                padding=(int((self.h - 1) // 2), int((self.w - 1) // 2)),
            )
        else:
            x = F.conv2d(
                x,
                f1,
                groups=batch_size,
                padding=(
                    int((self.h - 1) * self.dilate_height_rate // 2),
                    int((self.w - 1) * self.dilate_width_rate // 2),
                ),
                dilation=(self.dilate_height_rate, self.dilate_width_rate),
            )
        x = x.reshape(batch_size, self.output_channel, self.reshape_H, self.reshape_W)
        x = self.bn(x)
        x = self.se(x)
        return x


class MSRSCImprove(KBCModel):
    def __init__(
        self,
        num_ent,
        num_rel,
        embedding_dim=300,
        input_drop=0.4,
        hidden_drop=0.3,
        feature_map_drop=0.3,
        k_w=10,
        k_h=20,
        output_channel=20,
        filter_size_list=None,
        active_fn="relu",
        init_fn="xavier_normal",
        relation_stats=None,
        use_router=True,
        router_hidden=64,
        router_temperature=1.0,
        use_anchor=True,
        anchor_topk=8,
        anchor_alpha=0.25,
        anchor_dropout=0.1,
        device=None,
    ):
        super(MSRSCImprove, self).__init__()
        if filter_size_list is None:
            filter_size_list = [(1, 5), (3, 3), (1, 9)]

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_ent, embedding_dim),
                nn.Embedding(num_rel, embedding_dim),
            ]
        )
        self.emb_ent = self.embeddings[0]
        self.emb_rel = self.embeddings[1]
        self.embedding_dim = embedding_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.perm = 1
        self.k_w = k_w
        self.k_h = k_h
        self.reshape_H = self.k_w * 2
        self.reshape_W = self.k_h
        self.in_channel = 1
        self.active_fn = self.get_active_fn(active_fn)
        self.init_fn = self.get_init_fn(init_fn)

        if self.k_w * self.k_h > self.embedding_dim:
            raise ValueError("k_w * k_h must be <= embedding_dim.")

        self.register_buffer("chequer_perm", self.get_chequer_perm())

        self.num_filters = len(filter_size_list)
        if isinstance(output_channel, int):
            self.output_channels_list = [output_channel] * self.num_filters
        else:
            self.output_channels_list = output_channel
        if len(self.output_channels_list) != self.num_filters:
            raise ValueError("output_channel length must match filter_size_list length.")

        self.conv_layers = nn.ModuleList()
        for out_ch, filter_size in zip(self.output_channels_list, filter_size_list):
            self.conv_layers.append(
                RelationSpecificConv(
                    num_rel=num_rel,
                    in_channel=self.in_channel,
                    output_channel=out_ch,
                    filter_size=filter_size,
                    reshape_H=self.reshape_H,
                    reshape_W=self.reshape_W,
                    init_fn=self.init_fn,
                    emb_dim=self.embedding_dim,
                )
            )

        total_channel = sum(self.output_channels_list)
        self.input_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_map_drop = nn.Dropout2d(feature_map_drop)
        self.bn0 = nn.BatchNorm2d(self.in_channel)
        self.bn1 = nn.BatchNorm2d(total_channel)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(self.reshape_H * self.reshape_W * total_channel, embedding_dim)
        self.register_parameter("b", Parameter(torch.zeros(num_ent)))

        if relation_stats is None:
            relation_stats_tensor = torch.empty(num_rel, 0, dtype=torch.float32)
        else:
            relation_stats_tensor = torch.as_tensor(relation_stats, dtype=torch.float32)
            if relation_stats_tensor.shape[0] != num_rel:
                raise ValueError("relation_stats first dimension must equal num_rel.")
        self.register_buffer("relation_stats", relation_stats_tensor)

        self.use_router = use_router
        self.router_temperature = max(router_temperature, 1e-6)
        router_input_dim = embedding_dim + relation_stats_tensor.shape[1]
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, self.num_filters),
        )

        self.use_anchor = use_anchor
        self.anchor_topk = anchor_topk
        self.anchor_alpha = anchor_alpha
        self.anchor_dropout = nn.Dropout(anchor_dropout)
        self.anchor_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )
        self.anchor_norm = nn.LayerNorm(embedding_dim)
        self.anchor_resolver = None
        self.last_router_gate = None

    def set_anchor_resolver(self, resolver):
        self.anchor_resolver = resolver

    def init(self):
        self.init_fn(self.emb_ent.weight.data)
        self.init_fn(self.emb_rel.weight.data)
        self.init_fn(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for conv_layer in self.conv_layers:
            conv_layer.init_weights()
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                self.init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.anchor_gate.modules():
            if isinstance(module, nn.Linear):
                self.init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_active_fn(self, active_fn_name):
        fn_map = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "silu": F.silu,
            "softplus": F.softplus,
            "gelu": F.gelu,
            "elu": F.elu,
            "selu": F.selu,
        }
        if active_fn_name not in fn_map:
            raise ValueError("Unsupported activation function: {}".format(active_fn_name))
        return fn_map[active_fn_name]

    def get_init_fn(self, init_fn_name):
        fn_map = {
            "xavier_normal": nn.init.xavier_normal_,
            "xavier_uniform": nn.init.xavier_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
        }
        return fn_map.get(init_fn_name, nn.init.xavier_normal_)

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.k_h):
                for _ in range(self.k_w):
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
        return torch.as_tensor(np.int32(comb_idx), dtype=torch.long)

    def forward(self, x, anchor_ids=None, anchor_mask=None):
        x = x.to(self.device)
        e1 = x[:, 0].long()
        rel = x[:, 1].long()
        e2 = x[:, 2].long()

        z, e1_embedded, rel_embedded = self._calculate_embedding(e1, rel)
        if self.use_anchor:
            z = self._apply_anchor(z, rel_embedded, x, anchor_ids, anchor_mask)

        e2_embedded = self.emb_ent(e2)
        pred = torch.mm(z, self.emb_ent.weight.transpose(1, 0))
        pred = pred + self.b.expand_as(pred)
        return pred, [(e1_embedded, rel_embedded, e2_embedded)]

    def _calculate_embedding(self, e1, rel):
        e1_embedded = self.emb_ent(e1)
        rel_embedded = self.emb_rel(rel)
        comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.perm, self.reshape_H, self.reshape_W))
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
        x = x.permute(1, 0, 2, 3)

        outputs = [conv(e1_embedded, x, rel, rel_embedded) for conv in self.conv_layers]
        if self.use_router:
            outputs = self._apply_router(outputs, rel, rel_embedded)

        x = torch.cat(outputs, dim=1)
        x = self.bn1(x)
        x = self.active_fn(x)
        x = self.feature_map_drop(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.active_fn(x)
        return x, e1_embedded, rel_embedded

    def _apply_router(self, outputs, rel, rel_embedded):
        if self.relation_stats.shape[1] > 0:
            stats = self.relation_stats[rel].to(rel_embedded.dtype)
            router_input = torch.cat([rel_embedded, stats], dim=1)
        else:
            router_input = rel_embedded
        gate = self.router(router_input) / self.router_temperature
        gate = torch.softmax(gate, dim=1)
        self.last_router_gate = gate.detach()
        return [
            output * gate[:, i].view(-1, 1, 1, 1)
            for i, output in enumerate(outputs)
        ]

    def _apply_anchor(self, z, rel_embedded, batch, anchor_ids=None, anchor_mask=None):
        if self.anchor_topk <= 0:
            return z
        if anchor_ids is None and self.anchor_resolver is not None:
            anchor_ids, anchor_mask = self.anchor_resolver(
                batch, self.anchor_topk, z.device, exclude_target=True
            )
        if anchor_ids is None or anchor_mask is None or anchor_ids.numel() == 0:
            return z

        safe_ids = anchor_ids.clamp_min(0)
        anchor_emb = self.emb_ent(safe_ids)
        valid = anchor_mask.to(anchor_emb.dtype)
        attn_logits = torch.sum(anchor_emb * z.unsqueeze(1), dim=-1) / math.sqrt(self.embedding_dim)
        attn_logits = attn_logits.masked_fill(~anchor_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1) * valid
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        proto = torch.sum(attn.unsqueeze(-1) * anchor_emb, dim=1)

        has_anchor = anchor_mask.any(dim=1, keepdim=True).to(z.dtype)
        proto = proto * has_anchor
        gate = self.anchor_gate(torch.cat([z, rel_embedded, proto], dim=1))
        z = self.anchor_norm(z + self.anchor_alpha * gate * self.anchor_dropout(proto))
        return z


MSDCSE = MSRSCImprove
