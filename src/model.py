import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch.utils.checkpoint import checkpoint

import math
from typing import Literal


class CustomLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        sigma = math.sqrt(2 / (in_features+out_features))

        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
              torch.empty((out_features, in_features)), std=sigma, a=-3*sigma, b=3*sigma
            )
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... hidden, intermediate hidden -> ... intermediate") 


class CustomEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        sigma = 1.0
        self.Embedding = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim)), std=sigma, a=-3*sigma, b=3*sigma
                )
        )


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.Embedding[token_ids]


def factory_make_linear(in_features: int, out_features: int, is_custom: bool = False) -> nn.Module:
        """Factory function to create Linear layers in nn"""
        if is_custom:
                return CustomLinear(in_features, out_features)
        return nn.Linear(in_features, out_features, bias=False)


class RMSLayerNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.g = torch.nn.Parameter(
            torch.ones(self.d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.g).to(in_type)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, is_custom: bool = False):
        super().__init__()

        self.up_projection = factory_make_linear(d_model, d_ff, is_custom)
        self.gate_projection = factory_make_linear(d_model, d_ff, is_custom)
        self.down_projection = factory_make_linear(d_ff, d_model, is_custom)

        self.act_function = self.silu if is_custom else F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_projection(self.act_function(self.gate_projection(x)) * self.up_projection(x))

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RotaryEmbedding(torch.nn.Module):
    """RoPE with Eleuther splitting halves layout"""
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k)) # [d_k/2]
        pos = torch.arange(max_seq_len).float() # [S]
        freqs = einsum(pos, inv_freq, "s, f -> s f") # [S, d_k/2]
        emb = repeat(freqs, "s f -> s (two f)", two=2) # [S, d_k] 

        sign = torch.cat([-torch.ones(d_k // 2), torch.ones(d_k // 2)]) # [-1, 1]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin_sign", emb.sin() * sign, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x (batch, heads, seq_len, d_k)
        seq_len = x.shape[-2]
        
        x_dtype = x.dtype
        assert offset + seq_len <= self.max_seq_len, "sequence exceeds max_seq_len"

        cos = rearrange(self.cos[offset:offset + seq_len], "s d -> 1 1 s d")
        sin_sign = rearrange(self.sin_sign[offset:offset + seq_len], "s d -> 1 1 s d")

        x1, x2  = x[..., : self.d_k // 2], x[..., self.d_k // 2 :]
        swapped = torch.cat([x2, x1], dim=-1)
        return (x.float() * cos + swapped.float() * sin_sign).to(x_dtype)


def custom_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """custom softmax with some numerical stability""" 
    exponentiated_x = torch.exp(x - torch.amax(x, dim=dim, keepdim=True))
    return exponentiated_x / torch.sum(exponentiated_x, dim=dim, keepdim=True)


def custom_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """Scaled dot product attention implementation
        Args:
        Q, K, V - expects (..., num_heads, seq_len, head_dim)
        mask: bool/float tensor, True/1.0 where attention is allowed - lower triangle for causal
        Returns:
            Output tensor of shape (..., num_heads, seq_len, head_dim).
    """
    # q/k/v = (batch, heads, seq_len, d_k)
    d_k =   K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf")) # 1s in lower triangle, get scores from them

    attention_weights = custom_softmax(attention_scores, dim=-1)

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v") # b x h x s x d_v


class MultiHeadAttention(torch.nn.Module):
    """
    Causal MultiHeadAttention with RoPE
    attention_impl is used to compare naive vs sdpa
    - 'sdpa':   F.scaled_dot_product_attention, memory-efficient on T4, but not FlashAttention
    - 'eager':  einsum-based custom_sdpa(), materializes (S, S) scores, O(S^2) memory growth
    """
    def __init__(self, 
                 d_model: int,
                 num_heads: int, 
                 positional_encoder: RotaryEmbedding, 
                 is_custom: bool = False, 
                 attention_impl: str = "sdpa"):
        super().__init__()

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_q, self.d_v = self.d_k, self.d_k
        self.attention_impl = attention_impl

        self.Q_proj = factory_make_linear(d_model, d_model, is_custom)
        self.K_proj = factory_make_linear(d_model, d_model, is_custom)
        self.V_proj = factory_make_linear(d_model, d_model, is_custom)

        self.O_proj = factory_make_linear(d_model, d_model, is_custom)

        self.positional_encoder = positional_encoder

        self._mask_cache = None


    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        mask = rearrange(
            torch.tril(
                torch.ones(seq_len, seq_len), diagonal=0).bool(), 
                "seq seq1 -> 1 1 seq seq1").to(device)
        self._mask_cache = (key, mask)
        return mask
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        Q, K, V = self.Q_proj(x), self.K_proj(x), self.V_proj(x)
        seq_len = x.shape[-2]

        Q = rearrange(Q, "... s (n_heads d_q) -> ... n_heads s d_q", n_heads=self.num_heads)
        K = rearrange(K, "... s (n_heads d_q) -> ... n_heads s d_q", n_heads=self.num_heads)
        V = rearrange(V, "... s (n_heads d_q) -> ... n_heads s d_q", n_heads=self.num_heads)
      
        # RoPE
        Q = self.positional_encoder(Q)
        K = self.positional_encoder(K)

        if self.attention_impl == "sdpa":
            attention_hook = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        elif self.attention_impl == "eager":
            mask = self._causal_mask(seq_len, Q.device)
            attention_hook = custom_sdpa(Q, K, V, mask)
        else:
            raise ValueError(f"Unknown attention_impl: {self.attention_impl}")
        
        attention_hook = rearrange(attention_hook, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        return self.O_proj(attention_hook) 


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, positional_encoder, is_custom: bool = True, attention_impl: str = "sdpa"):
        super().__init__()
        self.SwiGLU = SwiGLU(d_model, d_ff, is_custom)
        self.MHA = MultiHeadAttention(d_model, num_heads, positional_encoder, is_custom, attention_impl)
        
        self.pre_norm_attn = RMSLayerNorm(d_model)
        self.pre_norm_ffn = RMSLayerNorm(d_model) 


    def _attention_comp(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.MHA(self.pre_norm_attn(x))
    

    def forward(self, x:torch.Tensor, grad_checkpoint_strat: str = "none") -> torch.Tensor:
        use_selective_gp = (grad_checkpoint_strat == "selective" and self.training)
        
        if use_selective_gp:
            x = checkpoint(self._attention_comp, x, use_reentrant=False)
        else:
            x = self._attention_comp(x)
        x = x + self.SwiGLU(self.pre_norm_ffn(x))
        return x
  
class GPTLM(torch.nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 vocab_size: int, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len: int, 
                 theta: float = 10000.0, 
                 is_custom: bool = False, 
                 attention_impl: str = "sdpa"):
        super().__init__()

        self.positional_encoder = RotaryEmbedding(d_model // num_heads, max_seq_len, theta)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, self.positional_encoder, is_custom, attention_impl) for _ in range(num_layers)])

        self.final_layer_norm = RMSLayerNorm(d_model)
        
        if is_custom:
            self.embedding = CustomEmbedding(vocab_size, d_model)
            self.lm_head = CustomLinear(in_features=d_model, out_features=vocab_size)
        else:
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.lm_head = torch.nn.Linear(in_features=d_model, out_features=vocab_size, bias=False)


    def forward(self, token_ids: torch.Tensor, 
                grad_checkpoint_strat: str = "none") -> torch.Tensor:
        # token_ids - b_s x seq_len, only token_ids
        x = self.embedding(token_ids)

        is_training = self.training and grad_checkpoint_strat != "none"

        for layer in self.layers:
            if is_training and grad_checkpoint_strat == "full":
                x = checkpoint(layer, x, use_reentrant=False)
            elif is_training and grad_checkpoint_strat == "selective":
                x = layer(x, grad_checkpoint_strat)
            else:
                x = layer(x)       
        x = self.final_layer_norm(x)

        return self.lm_head(x)
    

    def get_num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def get_model_size_in_gb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 ** 3
    