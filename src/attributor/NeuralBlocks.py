import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class DecoderCrossBlock(nn.Module):
    def __init__(self, n_embed, n_heads, query_pos_emb=None, key_pos_emb=None):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads

        self.query_pos_emb = query_pos_emb
        self.key_pos_emb = key_pos_emb
       
        self.decoders = nn.ModuleList([
            TransformerBlock(n_embd=self.n_embed, n_head=self.n_heads, is_causal=True,
                              query_pos_emb= self.query_pos_emb, key_pos_emb = self.key_pos_emb ) for _ in range(2)
        ])

        #self.source_cross_att_norm = nn.LayerNorm(self.n_embed)
        self.target_cross_att_norm = nn.LayerNorm(self.n_embed)
        self.cross_att = CrossAttention(n_head=self.n_heads, n_embd=self.n_embed, is_value=True,
                                         query_pos_emb=self.query_pos_emb, key_pos_emb=self.key_pos_emb)
        
    def forward(self, precomputed_source, source_mask, target, target_mask):

        for block in self.decoders:
            target = block(target, attention_mask=target_mask)

        target = self.target_cross_att_norm(target)
        #precomputed_source = self.source_cross_att_norm(precomputed_source)
        result = self.cross_att(target, precomputed_source, target_mask, source_mask)
        return result
    
class HeadGateMLP(nn.Module):
    def __init__(self, n_embed, n_heads, hidden=128, dropout=0.1):
        super().__init__()
        self.ln   = nn.LayerNorm(n_embed)
        self.fc1  = nn.Linear(n_embed, hidden)
        self.fc2  = nn.Linear(hidden, n_heads)
        self.do   = nn.Dropout(dropout)

    def forward(self, h_ctx):
        x = self.ln(h_ctx)
        x = F.gelu(self.fc1(x))
        x = self.do(x)
        logits = self.fc2(x)                
        alpha  = logits.softmax(dim=-1)     
        return alpha

class ConvLengthPredictor(nn.Module):
    def __init__(self, n_embd, target_len_max, k=5):
        super().__init__()

        self.n_embd = n_embd
        self.target_len_max = target_len_max
        self.conv1 = nn.Conv1d(n_embd, n_embd, kernel_size=k, padding=k//2)
        self.bn1   = nn.BatchNorm1d(n_embd)
        self.conv2 = nn.Conv1d(n_embd, n_embd, kernel_size=k, padding=k//2)
        self.bn2   = nn.BatchNorm1d(n_embd)

        self.fc1   = nn.Linear(n_embd * 2 + 1, n_embd // 2)
        self.fc2   = nn.Linear(n_embd // 2, target_len_max)

    def forward(self, x, x_mask, true_length=None):
        # H: [B, T, C]
        #conv to extract features
        x = x * x_mask.unsqueeze(-1)
        x = x.transpose(1, 2)             #  [B, C, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  #  [B, ch, T]

        mask_2d = x_mask.unsqueeze(1).expand(-1,x.size(1),-1).bool() # [B, ch, T]
        #extracting features
        #masked max pooling
        neg_inf = torch.finfo(x.dtype).min
        masked = x.masked_fill(~mask_2d.bool(), neg_inf)  
        v_max = masked.max(dim=2).values 
        #masked average pooling
        valid_counts = mask_2d.sum(dim=-1, keepdim=True).squeeze(-1)  # [B, ch]        
        sum_pooled   = (x * mask_2d).sum(-1)
        v_avg        = sum_pooled / valid_counts  # [B, ch]
        #normalized source_length
        v_src_len = (x_mask.sum(dim=-1) / self.target_len_max).unsqueeze(-1) # [B, 1]

        #concatenate features
        v = torch.cat([v_max, v_avg, v_src_len], dim=1)  # [B, 2*ch]

        # MLP classification
        h = F.relu(self.fc1(v)) # [B, d_model // 2]
        logits = self.fc2(h) # [B, target_len_max]

        loss = None
        if true_length is not None:
            loss = F.cross_entropy(logits, true_length)
        
        return logits.softmax(-1).argmax(-1), loss
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, is_causal=False, query_pos_emb=None, key_pos_emb=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head=n_head, n_embd=n_embd, is_causal=is_causal, query_pos_emb=query_pos_emb, key_pos_emb=key_pos_emb)
        self.do1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.do2 = nn.Dropout(0.1)
    
    def forward(self, x, attention_mask=None):
        # Pre-LN + residual
        x = x + self.do1(self.attention(self.norm1(x), attention_mask=attention_mask))
        x = x + self.do2(self.mlp(self.norm2(x)))
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, query_pos_emb=None, key_pos_emb=None):
        super().__init__()
        self.norm_t = nn.LayerNorm(n_embd)
        self.norm_s = nn.LayerNorm(n_embd)
        self.attention = CrossAttention(n_head=n_head, n_embd=n_embd,
                                         query_pos_emb=query_pos_emb, key_pos_emb=key_pos_emb, is_value=True)
        self.mlp = MLP(n_embd)
    
    def forward(self, target, source, target_mask = None, source_mask = None):
        target = self.norm_t(target)
        source = self.norm_s(source)
        x = self.attention(target, source, target_mask=target_mask, source_mask=source_mask)
        x = self.mlp(x)
        return x

class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class SelfAttention(nn.Module):
    '''Self attention module to use in the encoder and decoder layers before cross-attention'''
    def __init__(self, n_head, n_embd, is_causal=True, query_pos_emb=None, key_pos_emb=None):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd

        self.query_pos_emb = query_pos_emb
        self.key_pos_emb = key_pos_emb

        assert n_embd % n_head == 0

        self.w_q  = nn.Linear(self.n_embd, self.n_embd)
        self.w_k = nn.Linear(self.n_embd, self.n_embd)
        self.w_v  = nn.Linear(self.n_embd, self.n_embd)
        self.output_proj = nn.Linear(self.n_embd, self.n_embd)
        self.is_causal = is_causal


    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        
        if self.query_pos_emb is not None:
            q = self.w_q(x + self.query_pos_emb(torch.arange(T, device=x.device)))
        else:
            q = self.w_q(x)
        if self.key_pos_emb is not None:
            k = self.w_k(x + self.key_pos_emb(torch.arange(T, device=x.device)))
        else:
            k = self.w_k(x)
        v = self.w_v(x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        result_mask = None

        # Create causal mask (lower triangular)
        if self.is_causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            causal_mask = causal_mask[None, None, :, :].expand(B, self.n_head, T, T)
            result_mask = causal_mask
        
        # if attention_mask is provided, combine it with the causal mask
        if attention_mask is not None:
            # attention_mask shape: (B, T) where 1 = real token, 0 = padding.
            # The mask is square as it is self-attention
            if attention_mask.dim() == 2:
                query_mask = attention_mask[:, :, None]  # (B, T, 1) 
                key_mask = attention_mask[:, None, :]    # (B, 1, T) 
                padding_mask = query_mask & key_mask     # (B, T, T)
            elif attention_mask.dim() == 3:
                padding_mask = attention_mask            # (B, T, T)
            else:
                raise ValueError("attention_mask must have 2 or 3 dimensions")
            padding_mask = padding_mask[:, None, :, :].expand(B, self.n_head, T, T).bool()  # (B, n_head, T, T)
            result_mask = padding_mask & result_mask if result_mask is not None else padding_mask

        y = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        y = y.masked_fill(~result_mask, -1e10)
        y = F.softmax(y, dim=-1)
        y = y @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.output_proj(y)
        return y

class CrossAttention(nn.Module):
    def __init__(self, n_head, n_embd, is_value=False, avg_heads=True, log_softmax=False, query_pos_emb=None, key_pos_emb=None):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.avg_heads = avg_heads
        self.log_softmax = log_softmax

        self.query_pos_emb = query_pos_emb
        self.key_pos_emb = key_pos_emb

        assert n_embd % n_head == 0

        #We don't need value 
        self.w_k = nn.Linear(self.n_embd, self.n_embd)
        self.w_q  = nn.Linear(self.n_embd, self.n_embd)
        if is_value:
            self.w_v = nn.Linear(self.n_embd, self.n_embd)
        else:
            self.w_v = None
        self.output_proj = nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, target, source, target_mask, source_mask):
        B, T_t, C = target.size()
        T_s = source.size(1)

        #if target or source mask is not provided, we assume that all tokens are valid
        if target_mask is None:
            target_mask = torch.ones(B, T_t, dtype=torch.bool, device=target.device)
        if source_mask is None:
            source_mask = torch.ones(B, T_s, dtype=torch.bool, device=source.device)

        #actually not important anymore
        #assert target.size() == source.size(), "x and context must have the same shape"

        if self.key_pos_emb is not None:
            k = self.w_k(source + self.key_pos_emb(torch.arange(T_s, device=source.device)))
        else:
            k = self.w_k(source)
        if self.query_pos_emb is not None:
            q = self.w_q(target + self.query_pos_emb(torch.arange(T_t, device=target.device)))
        else:
            q = self.w_q(target)

        k = k.view(B, T_s, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_t, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #calculate rectangular padding mask depending on the masks of the target and source
        query_mask = target_mask[:, :, None]  # (B, T, 1) 
        key_mask = source_mask[:, None, :]    # (B, 1, T) 
        padding_mask = query_mask & key_mask  # (B, T, T)
        result_mask = padding_mask[:, None, :, :].expand(B, self.n_head, T_t, T_s).bool()  # (B, n_head, T, T)

        # manual attention as we need raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        attn_scores = attn_scores.masked_fill(~result_mask, -1e10)
        if self.log_softmax:
            attn_probs = attn_scores #F.log_softmax(attn_scores, dim=-1)
        else:
            attn_probs = F.softmax(attn_scores, dim=-1)
        result = attn_probs

        # if value is not None, we provide resulting embeddings, otherwise we return the mean of the attention probabilities
        if self.w_v is not None:
            v = self.w_v(source)
            v = v.view(B, T_s, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            result = attn_probs @ v
            result = result.transpose(1, 2).contiguous().view(B, T_t, C)
        else:
            if self.avg_heads:
                result = attn_probs.mean(dim=1)
        
        return result

class TAConfig:
    def __init__(self, n_window, n_embed, n_heads, n_vocabulary, pad_token_id=None):
        self.n_window = n_window
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_vocabulary = n_vocabulary 
        self.pad_token_id = pad_token_id

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, n_pos: int, dim: int):
        super().__init__()
        pe = torch.zeros(n_pos, dim)
        position = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
        #div term is 1/10000^(2i/dim) for i in [0, 1, ..., dim-1] but algbraically transformed
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  #(1, n_positions, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]