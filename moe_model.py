"""
MoE GPT Model - HuggingFace kompatibel
Basiert auf nanoMoE und dem Blog Post
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from moe_config import MoEGPTConfig
from moe_layers import MoELayer


@dataclass
class MoECausalLMOutput(CausalLMOutputWithPast):
    """
    Erweiterte Output Klasse mit MoE-spezifischen Losses
    """

    aux_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to input tensor.

    Args:
        x: Input tensor of shape [B, H, T, D]
        freqs_cos: Cosine frequencies of shape [T, D//2]
        freqs_sin: Sine frequencies of shape [T, D//2]

    Returns:
        Tensor with RoPE applied
    """
    # Reshape x to separate real and imaginary parts for rotation
    # x: [B, H, T, D] -> [B, H, T, D//2, 2]
    x_complex = x.float().reshape(*x.shape[:-1], -1, 2)

    # Apply rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    x_rot_real = x_complex[..., 0] * freqs_cos - x_complex[..., 1] * freqs_sin
    x_rot_imag = x_complex[..., 0] * freqs_sin + x_complex[..., 1] * freqs_cos

    # Stack back together and flatten
    x_out = torch.stack([x_rot_real, x_rot_imag], dim=-1)
    x_out = x_out.flatten(-2)

    return x_out.type_as(x)


def precompute_freqs_rope(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes RoPE frequencies.

    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        theta: RoPE theta parameter (base for frequency calculation)

    Returns:
        Tuple of (freqs_cos, freqs_sin) tensors of shape [max_seq_len, dim//2]
    """
    # Compute frequencies for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute outer product: [max_seq_len, dim//2]
    freqs = torch.outer(t, freqs)

    # Compute cos and sin
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Position Embeddings (RoPE).
    Uses PyTorch SDPA for optimized performance.
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value für alle Heads gleichzeitig
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output Projektion
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_rope(
            dim=self.head_dim,
            max_seq_len=config.n_positions,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Q, K, V berechnen
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape für Multi-Head
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, self.freqs_cos[:T], self.freqs_sin[:T])
        k = apply_rotary_emb(k, self.freqs_cos[:T], self.freqs_sin[:T])

        # Use PyTorch SDPA (Scaled Dot Product Attention) - optimized!
        # SDPA handles causal masking, dropout, and is memory efficient
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Causal mask handled by is_causal
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True  # Efficient causal masking
        )  # [B, H, T, d]

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output Projektion
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """
    Standard Feed-Forward Network (für nicht-MoE Layers)
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        if config.activation_function == "gelu":
            self.activation = nn.GELU()
        elif config.activation_function == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unbekannte Aktivierung: {config.activation_function}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block (Attention + MLP)
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MoETransformerBlock(nn.Module):
    """
    MoE Transformer Block (Attention + MoE Layer)
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Capacity Factor abhängig von Training/Eval
        self.moe = MoELayer(
            d_model=config.n_embd,
            n_experts=config.n_experts,
            n_experts_active=config.n_experts_active,
            use_noisy_gating=config.use_noisy_gating,
            capacity_factor=config.capacity_factor,
            bias=config.bias,
            dropout=config.dropout,
            activation=config.activation_function,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention
        x = x + self.attn(self.ln_1(x))

        # MoE Layer
        moe_out, aux_loss, router_z_loss = self.moe(self.ln_2(x))
        x = x + moe_out

        return x, aux_loss, router_z_loss


class MoEGPTPreTrainedModel(PreTrainedModel):
    """
    Base Klasse für MoE GPT mit HuggingFace PreTrainedModel
    """

    config_class = MoEGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        Weight Initialization nach ST-MoE (Zoph et al. 2022)
        Truncated Normal mit reduzierter Std für MoE Stabilität
        """
        if isinstance(module, nn.Linear):
            # Fan-in Initialization
            fan_in = module.weight.shape[-1]
            std = (self.config.initializer_range / fan_in) ** 0.5

            torch.nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.Parameter):
            # Für Expert Parameter
            fan_in = module.shape[-1] if len(module.shape) >= 2 else module.shape[0]
            std = (self.config.initializer_range / fan_in) ** 0.5

            torch.nn.init.trunc_normal_(
                module,
                mean=0.0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )


class MoEGPTModel(MoEGPTPreTrainedModel):
    """
    MoE GPT Model (ohne LM Head)
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False  # Für HF Gradient Checkpointing Support

        # Token Embeddings only (RoPE handles positions)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks (gemischt: Standard + MoE)
        self.h = nn.ModuleList()
        for i in range(config.n_layer):
            if i % config.moe_layer_frequency == 0:
                # MoE Block
                self.h.append(MoETransformerBlock(config))
            else:
                # Standard Block
                self.h.append(TransformerBlock(config))

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = input_ids.device
        b, t = input_ids.size()

        assert t <= self.config.n_positions, f"Sequenz zu lang: {t} > {self.config.n_positions}"

        # Token Embeddings only (RoPE in attention layers)
        tok_emb = self.wte(input_ids)  # [B, T, n_embd]
        x = self.drop(tok_emb)

        # Sammle Auxiliary Losses
        total_aux_loss = 0.0
        total_router_z_loss = 0.0

        # Durch alle Blocks
        for block in self.h:
            if isinstance(block, MoETransformerBlock):
                if self.gradient_checkpointing and self.training:
                    # Gradient Checkpointing für MoE Blocks
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    x, aux_loss, router_z_loss = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        use_reentrant=False
                    )
                else:
                    x, aux_loss, router_z_loss = block(x)
                total_aux_loss = total_aux_loss + aux_loss
                total_router_z_loss = total_router_z_loss + router_z_loss
            else:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        block,
                        x,
                        use_reentrant=False
                    )
                else:
                    x = block(x)

        x = self.ln_f(x)

        return x, total_aux_loss, total_router_z_loss


class MoEGPTForCausalLM(MoEGPTPreTrainedModel, GenerationMixin):
    """
    MoE GPT mit Language Modeling Head (für Pretraining)
    Erbt von GenerationMixin für .generate() Support
    """

    # Teile HuggingFace mit, welche Weights geteilt sind
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MoEGPTConfig):
        super().__init__(config)
        self.transformer = MoEGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (LM Head teilt Gewichte mit Token Embedding)
        self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        """Für HuggingFace Weight Tying"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Für HuggingFace Weight Tying"""
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """Für HuggingFace Weight Tying"""
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        """Für HuggingFace Weight Tying"""
        self.transformer.wte = new_embeddings

    def tie_weights(self):
        """
        Tie lm_head weights to input embeddings (weight tying)
        Called after loading checkpoint to fix missing lm_head.weight
        """
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward durch Transformer
        hidden_states, aux_loss, router_z_loss = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # LM Head
        if labels is not None:
            # Training: nur letzte Position für jede Sequenz
            logits = self.lm_head(hidden_states)
        else:
            # Inference: nur letzte Position
            logits = self.lm_head(hidden_states[:, [-1], :])

        # Loss berechnen
        loss = None
        if labels is not None:
            # Shift für next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross Entropy Loss
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Auxiliary Losses hinzufügen
            loss = lm_loss
            if self.training:
                loss = loss + self.config.aux_loss_alpha * aux_loss
                loss = loss + self.config.router_z_loss_alpha * router_z_loss

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MoECausalLMOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss if self.training else None,
            router_z_loss=router_z_loss if self.training else None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Für HuggingFace generate() Funktion"""
        return {"input_ids": input_ids}
