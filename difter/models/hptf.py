"""The final HPTF traffic-window encoder.

The implementation keeps the actual six-stage path: token/context encoding,
auxiliary timing and length fusion, intra-packet field modeling, inter-packet
temporal modeling, multi-branch pooling, and the final 768-dimensional window
representation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyLayerNorm(nn.Module):
    """Layer normalization used by the pretrained traffic encoder."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(-1, keepdim=True)
        std = value.std(-1, keepdim=True)
        return self.gamma * (value - mean) / (std + self.eps) + self.beta


class WordPositionSegmentEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_length: int, dropout: float):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.segment_embedding = nn.Embedding(3, hidden_size)
        self.layer_norm = LegacyLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        value = self.word_embedding(token_ids) + self.position_embedding(positions) + self.segment_embedding(segments)
        return self.dropout(self.layer_norm(value))


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.heads_num = heads
        self.per_head_size = hidden_size // heads
        self.inner_hidden_size = hidden_size
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = query.shape
        query, key, value = [layer(tensor).view(batch, -1, self.heads_num, self.per_head_size).transpose(1, 2)
                             for layer, tensor in zip(self.linear_layers, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(self.per_head_size))
        probabilities = F.softmax(scores + mask, dim=-1)
        context = torch.matmul(self.dropout(probabilities), value).transpose(1, 2).contiguous()
        return self.final_linear(context.view(batch, length, self.inner_hidden_size))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, heads: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadedAttention(hidden_size, heads, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, feedforward_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_1 = LegacyLayerNorm(hidden_size)
        self.layer_norm_2 = LegacyLayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        return self.layer_norm_2(output + inter)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, feedforward_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)

    def forward(self, value):
        return self.linear_2(F.gelu(self.linear_1(value)))


class ContextEncoder(nn.Module):
    def __init__(self, hidden_size: int, heads: int, feedforward_size: int, layers: int, dropout: float):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerLayer(hidden_size, heads, feedforward_size, dropout) for _ in range(layers)
        ])

    def forward(self, embedded: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        length = embedded.size(1)
        visible = (segments > 0).unsqueeze(1).repeat(1, length, 1).unsqueeze(1).float()
        mask = (1.0 - visible) * -10000.0
        hidden = embedded
        for layer in self.transformer:
            hidden = layer(hidden, mask)
        return hidden


class HPTFEncoder(nn.Module):
    """Encode one traffic window into a flow-semantic representation."""

    def __init__(self, vocab_size: int = 60005, max_seq_length: int = 128,
                 hidden_size: int = 768, heads: int = 12, feedforward_size: int = 3072,
                 context_layers: int = 12, intra_packet_layers: int = 1,
                 inter_packet_layers: int = 2, field_slots: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_slots = field_slots
        self.embedding = WordPositionSegmentEmbedding(vocab_size, hidden_size, max_seq_length, dropout)
        self.encoder = ContextEncoder(hidden_size, heads, feedforward_size, context_layers, dropout)
        self.time_feature_proj = nn.Linear(1, hidden_size)
        self.length_feature_proj = nn.Linear(1, hidden_size)
        self.aux_feature_norm = nn.LayerNorm(hidden_size)
        self.aux_gate = nn.Linear(hidden_size * 2, hidden_size)
        field_layer = nn.TransformerEncoderLayer(hidden_size, heads, feedforward_size, dropout, "gelu", batch_first=True)
        self.field_encoder = nn.TransformerEncoder(field_layer, intra_packet_layers, enable_nested_tensor=False)
        self.field_pos_embedding = nn.Embedding(field_slots, hidden_size)
        self.field_attention = nn.Linear(hidden_size, 1)
        packet_layer = nn.TransformerEncoderLayer(hidden_size, heads, feedforward_size, dropout, "gelu", batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(packet_layer, inter_packet_layers, enable_nested_tensor=False)
        self.packet_pos_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.packet_attention = nn.Linear(hidden_size, 1)
        self.direct_token_attention = nn.Linear(hidden_size, 1)
        self.flow_stats_proj = nn.Sequential(nn.Linear(4, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size))
        self.cls_fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size))
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.output_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.feature_dim = hidden_size

    def _fuse_aux(self, embedded, segments, delta_ts, packet_length):
        time = self.time_feature_proj(delta_ts.unsqueeze(-1))
        length = self.length_feature_proj(packet_length.unsqueeze(-1))
        auxiliary = self.aux_feature_norm(time + length) * (segments > 0).unsqueeze(-1)
        gate = torch.sigmoid(self.aux_gate(torch.cat([embedded, auxiliary], dim=-1)))
        return embedded + gate * auxiliary

    @staticmethod
    def _flow_stats(segments, delta_ts, packet_length):
        valid = (segments > 0).float()
        denominator = valid.sum(1, keepdim=True).clamp_min(1.0)
        time_mean = (delta_ts * valid).sum(1, keepdim=True) / denominator
        length_mean = (packet_length * valid).sum(1, keepdim=True) / denominator
        time_var = (((delta_ts - time_mean) * valid) ** 2).sum(1, keepdim=True) / denominator
        length_var = (((packet_length - length_mean) * valid) ** 2).sum(1, keepdim=True) / denominator
        return torch.cat([time_mean, (time_var + 1e-12).sqrt(), length_mean, (length_var + 1e-12).sqrt()], 1)

    def _packet_table(self, features, segments, packet_start):
        batch, _, hidden = features.shape
        samples, max_packets = [], 1
        for row in range(batch):
            token_limit = int((segments[row] > 0).sum().item())
            starts = torch.nonzero(packet_start[row] > 0, as_tuple=False).flatten().tolist()
            starts = [position for position in starts if position < token_limit] or [0]
            packets = []
            for index, start in enumerate(starts):
                end = min(starts[index + 1], token_limit) if index + 1 < len(starts) else token_limit
                if end <= start:
                    continue
                count = min(end - start, self.field_slots)
                slot = features.new_zeros(self.field_slots, hidden)
                valid = torch.zeros(self.field_slots, dtype=torch.bool, device=features.device)
                slot[:count] = features[row, start:start + count]
                valid[:count] = True
                packets.append((slot, valid))
            if not packets:
                slot = features.new_zeros(self.field_slots, hidden)
                valid = torch.zeros(self.field_slots, dtype=torch.bool, device=features.device)
                slot[0], valid[0] = features[row, 0], True
                packets.append((slot, valid))
            samples.append(packets)
            max_packets = max(max_packets, len(packets))
        table = features.new_zeros(batch, max_packets, self.field_slots, hidden)
        field_valid = torch.zeros(batch, max_packets, self.field_slots, dtype=torch.bool, device=features.device)
        packet_valid = torch.zeros(batch, max_packets, dtype=torch.bool, device=features.device)
        for row, packets in enumerate(samples):
            for packet, (slot, valid) in enumerate(packets):
                table[row, packet], field_valid[row, packet], packet_valid[row, packet] = slot, valid, True
        return table, field_valid, packet_valid

    def forward_features(self, token_ids, segments, delta_ts, packet_length, packet_start):
        embedded = self._fuse_aux(self.embedding(token_ids, segments), segments, delta_ts, packet_length)
        contextual = self.encoder(embedded, segments)
        cls_pooled = contextual[:, 0]
        token_valid = segments > 0
        direct_logits = self.direct_token_attention(contextual).squeeze(-1).masked_fill(~token_valid, -1e4)
        direct_pooled = (contextual * F.softmax(direct_logits, -1).unsqueeze(-1)).sum(1)
        table, field_valid, packet_valid = self._packet_table(contextual, segments, packet_start)
        batch, packets, fields, hidden = table.shape
        table = table + self.field_pos_embedding(torch.arange(fields, device=table.device).view(1, 1, fields))
        field_input = table.reshape(batch * packets, fields, hidden)
        field_mask = ~field_valid.reshape(batch * packets, fields)
        empty = field_mask.all(1)
        if empty.any():
            field_mask[empty, 0] = False
        encoded_fields = self.field_encoder(field_input, src_key_padding_mask=field_mask)
        field_logits = self.field_attention(encoded_fields).squeeze(-1).masked_fill(~field_valid.reshape(batch * packets, fields), -1e4)
        packet_reps = (encoded_fields * F.softmax(field_logits, -1).unsqueeze(-1)).sum(1).reshape(batch, packets, hidden)
        packet_reps = packet_reps + self.packet_pos_embedding(torch.arange(packets, device=table.device).view(1, packets))
        encoded_packets = self.temporal_encoder(packet_reps, src_key_padding_mask=~packet_valid)
        packet_logits = self.packet_attention(encoded_packets).squeeze(-1).masked_fill(~packet_valid, -1e4)
        packet_pooled = (encoded_packets * F.softmax(packet_logits, -1).unsqueeze(-1)).sum(1)
        with torch.autocast(device_type=contextual.device.type, enabled=False):
            stats_pooled = self.flow_stats_proj(self._flow_stats(segments, delta_ts, packet_length).float())
        cls_base = packet_pooled + direct_pooled + stats_pooled
        cls_branch = self.cls_fusion(torch.cat([cls_base, cls_pooled], -1))
        pooled = self.fusion_norm(packet_pooled + direct_pooled + stats_pooled + cls_branch)
        return {"flow": torch.tanh(self.output_layer_1(pooled)), "packet": packet_pooled,
                "direct": direct_pooled, "stats": stats_pooled, "context": cls_branch}

    def forward(self, token_ids, segments, delta_ts, packet_length, packet_start):
        return self.forward_features(token_ids, segments, delta_ts, packet_length, packet_start)["flow"]
