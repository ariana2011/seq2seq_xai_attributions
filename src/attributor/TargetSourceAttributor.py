import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.stats import kendalltau
import numpy as np
from attributor.NeuralModule import NeuralModule
from attributor.NeuralBlocks import TransformerBlock, CrossAttention, TAConfig, HeadGateMLP, SinusoidalPositionalEmbedding, SelfAttention, MLP 

class TargetSourceAttributor(NeuralModule):
    def __init__(self, config: TAConfig):
        super().__init__(config)
        self.n_window = config.n_window
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.n_vocabulary = config.n_vocabulary
        self.pad_token_id = config.pad_token_id

        self.wte = nn.Embedding(self.n_vocabulary, self.n_embed)
        self.wpe_target = nn.Embedding(self.n_window, self.n_embed)
        self.wpe_source = nn.Embedding(self.n_window, self.n_embed)

        self.encoder_layers_source = nn.ModuleList([
            TransformerBlock(n_embd=self.n_embed, n_head=self.n_heads, is_causal=False) for _ in range(3)
        ])
       
        self.decoder_layers_target = nn.ModuleList([
            TransformerBlock(n_embd=self.n_embed, n_head=self.n_heads, is_causal=True) for _ in range(3)
        ])

        self.source_cross_att_norm = nn.LayerNorm(self.n_embed)
        self.target_cross_att_norm = nn.LayerNorm(self.n_embed)
        self.cross_att = CrossAttention(n_head=self.n_heads, n_embd=self.n_embed, is_value=False, avg_heads=False, log_softmax=True)

        self.head_gate_mlp = HeadGateMLP(n_embed=self.n_embed, n_heads=self.n_heads)

        
    def forward(self, target, source, target_mask, source_mask, attributions = None, do_stuff=False):

        #add BOS if it is not the first token
        if target[:,0].ne(self.pad_token_id).all():
            bos = torch.tensor([self.pad_token_id]).expand(target.shape[0], -1).to(target.device)  # BOS token
            target = torch.cat((bos, target), dim=1)[:, :self.n_window]

        if attributions is not None:
            assert attributions.shape[1] == source.shape[-1] and \
                attributions.shape[2] == target.shape[-1], \
                "Attributions is a matrix of shape (source_length, target_length) but got shape {}".format(attributions.shape)

        target = self.wte(target)  # (B, T, C)
        target = target + self.wpe_target(torch.arange(target.size(1), device=target.device))
        for layer in self.decoder_layers_target:
            target = layer(target, attention_mask=target_mask)

        source = self.wte(source)  # (B, T, C)
        source = source + self.wpe_source(torch.arange(source.size(1), device=source.device))
        for layer in self.encoder_layers_source:
            source = layer(source, attention_mask=source_mask)

        target = self.target_cross_att_norm(target)
        source = self.source_cross_att_norm(source)
        result_log = self.cross_att(target, source, target_mask, source_mask)

        head_weights = self.head_gate_mlp(target)
        sum_of_weightes_logs_across_heads  = (head_weights.permute(0,2,1).unsqueeze(-1) * result_log).sum(1)   # (B,T,S)
        result = torch.softmax(sum_of_weightes_logs_across_heads, dim=-1) 


        #transpose the attributions to make them compatible with the result with shape (target_length, source_length)
        if attributions is not None:
            attributions = attributions.transpose(-1,-2)
            # Create a mask for valid positions (non-padded) [B, source_len, target_len]
            expanded_source_mask = source_mask.unsqueeze(1).expand(-1, target.size(1), -1)
            expanded_target_mask = target_mask.unsqueeze(2).expand(-1, -1, source.size(1))
            valid_positions = expanded_target_mask & expanded_source_mask
            
            # recall your soldiers, we have normalization in the dataset already
            #assert torch.all((attributions.sum(dim=-1) - 1.0).abs() < 1e-3) and torch.all(attributions >= 0), \
            #    "Attributions rows must be a probability distribution"
            assert torch.all((result.sum(dim=-1) - 1.0).abs() < 1e-3) and torch.all(result >= 0), \
                "Result rows must be a probability distribution"
            
            
            # Apply smoothing to both distributions
            smooth_result = result.clone() + 1e-10
            smooth_attributions = attributions.float().clone() + 1e-10
            
            # Re-normalize to ensure they sum to 1
            smooth_result = smooth_result / smooth_result.sum(dim=-1, keepdim=True)
            smooth_attributions = smooth_attributions / smooth_attributions.sum(dim=-1, keepdim=True)
            
            #reverse Kl divergence calculation, i am too dead
            kl_div = (smooth_attributions * (torch.log(smooth_attributions) - torch.log(smooth_result))).sum(dim=-1)
            
            masked_kl_div = kl_div * target_mask
            loss = masked_kl_div.sum() / target_mask.sum()

            bonus = None
            if do_stuff:
                bonus = dict()
                bonus['avg_overlaps_k3'], bonus['avg_kts_k3'] = topk_overlap(result, attributions, source_mask.sum(dim=1), target_mask.sum(dim=1), k=3)
                result_masked = result * valid_positions.float()
                attributions_masked = attributions * valid_positions.float()
                bonus['frobenius'] = torch.norm(result_masked - attributions_masked, p='fro').item()
            
            return result, loss, ('KL-divergence', loss, bonus)
        
        return result, None, None

# @torch.no_grad()            
# def topk_overlap(result, attributions, src_len_batch, tgt_len_batch, k=5):
#     with torch.no_grad():
#         avg_overlaps = []
#         avg_kendall_taus = []
#         for i in range(result.size(0)):
#             src_len = src_len_batch[i].item()
#             tgt_len = tgt_len_batch[i].item()
#             k = min(k, src_len)
#             overlaps = []
#             kendall_taus = []
#             for j in range(tgt_len):
#                 result_topk = torch.topk(result[i,j,:src_len], k).indices
#                 attr_topk = torch.topk(attributions[i,:src_len,j], k).indices
#                 overlap = len(set(result_topk.cpu().numpy()).intersection(set(attr_topk.cpu().numpy())))
#                 overlap_ratio = overlap / k
#                 kendall_tau = kendalltau(result[i,j,:src_len].cpu().numpy(), attributions[i,:src_len,j].cpu().numpy())
#                 overlaps.append(overlap_ratio)
#                 kendall_taus.append(kendall_tau[0,1].item())
#             avg_overlap = sum(overlaps) / len(overlaps)
#             avg_kendall_tau = sum(kendall_taus) / len(kendall_taus)
#             avg_overlaps.append(avg_overlap)
#             avg_kendall_taus.append(avg_kendall_tau)
#         return sum(avg_overlaps) / len(avg_overlaps), sum(avg_kendall_taus) / len(avg_kendall_taus)

@torch.no_grad()
def topk_overlap(result, attributions, src_len_batch, tgt_len_batch, k=5):
    B, T, S = result.shape

    avg_overlaps = []
    avg_kendall_taus = []

    for i in range(B):
        src_len = int(src_len_batch[i].item())
        tgt_len = int(tgt_len_batch[i].item())
        if src_len == 0 or tgt_len == 0:
            continue

        k_eff = min(k, src_len)

        overlaps = []
        kendall_taus = []

        for j in range(tgt_len):
            # gold predicted rows restricted to valid source positions
            pred_row = result[i, j, :src_len]        
            gold_row = attributions[i, j, :src_len]  

            # top-k indices for pred & gold
            pred_topk_idx = torch.topk(pred_row, k_eff).indices
            gold_topk_idx = torch.topk(gold_row, k_eff).indices

            overlap = len(
                set(pred_topk_idx.cpu().tolist()) &
                set(gold_topk_idx.cpu().tolist())
            )
            overlap_ratio = overlap / float(k_eff)
            overlaps.append(overlap_ratio)

            # Take values at gold's top-k indices, in that order
            gold_vals = gold_row[gold_topk_idx].cpu().numpy()
            pred_vals = pred_row[gold_topk_idx].cpu().numpy()

            if k_eff < 2 or np.allclose(gold_vals, gold_vals[0]):
                tau = 0.0
            else:
                tau_val, _ = kendalltau(gold_vals, pred_vals)
                if np.isnan(tau_val):
                    tau = 0.0
                else:
                    tau = float(tau_val)

            kendall_taus.append(tau)

        if overlaps:
            avg_overlap = float(sum(overlaps) / len(overlaps))
            avg_kendall_tau = float(sum(kendall_taus) / len(kendall_taus))
            avg_overlaps.append(avg_overlap)
            avg_kendall_taus.append(avg_kendall_tau)

    if not avg_overlaps:
        return 0.0, 0.0

    return (
        float(sum(avg_overlaps) / len(avg_overlaps)),
        float(sum(avg_kendall_taus) / len(avg_kendall_taus)),
    )