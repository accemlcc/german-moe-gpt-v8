"""
MoE Layer Komponenten
Basierend auf dem nanoMoE Blog Post und HuggingFace Best Practices
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MoERouter(nn.Module):
    """
    Noisy Top-k Router für MoE.
    Routet Tokens zu den Top-k Experten basierend auf gelernten Wahrscheinlichkeiten.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_experts_active: int,
        use_noisy_gating: bool = True,
        capacity_factor: float = 1.25,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.n_experts_active = n_experts_active
        self.use_noisy_gating = use_noisy_gating
        self.capacity_factor = capacity_factor

        # Linear projections für Router (kein Bias, siehe Shazeer et al. 2017)
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.w_noise = nn.Linear(d_model, n_experts, bias=False) if use_noisy_gating else None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            expert_weights: Gewichte für jeden Experten [batch_size * seq_len, n_experts, capacity]
            expert_mask: Maske für verwendete Experten [batch_size * seq_len, n_experts, capacity]
            expert_batches: Batches für jeden Experten [n_experts, capacity, d_model]
            router_logits: Router Logits für z-loss [batch_size, seq_len, n_experts]
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Router läuft IMMER in FP32 für numerische Stabilität!
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()

            # Router Logits berechnen
            router_logits = self.w_gate(x_fp32)  # [B, T, n_experts]

            # Noisy Top-k Gating (optional)
            if self.use_noisy_gating and self.training:
                noise = F.softplus(self.w_noise(x_fp32))
                noise = noise * torch.randn_like(noise)
                router_logits = router_logits + noise

            # Top-k Experten auswählen
            top_k_logits, top_k_indices = router_logits.topk(
                self.n_experts_active, dim=-1
            )  # [B, T, K]

            # Softmax über alle Experten (nicht nur Top-k)
            router_probs = torch.full_like(router_logits, float("-inf"))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)  # [B, T, n_experts]

            # Expert Capacity berechnen
            capacity = self._compute_capacity(num_tokens)

            # Multi-hot Maske der gewählten Experten
            expert_mask = F.one_hot(
                top_k_indices, num_classes=self.n_experts
            )  # [B, T, K, n_experts]
            expert_mask = expert_mask.view(num_tokens, self.n_experts_active, self.n_experts)
            expert_mask = expert_mask.permute(1, 0, 2)  # [K, num_tokens, n_experts]

            # Position jedes Tokens im Expert Batch (cumsum für Top-1 first prioritization)
            expert_rank = expert_mask.reshape(
                self.n_experts_active * num_tokens, self.n_experts
            )
            expert_rank = torch.cumsum(expert_rank, dim=0) - 1
            expert_rank = expert_rank.reshape(
                self.n_experts_active, num_tokens, self.n_experts
            )

            # Tokens über Kapazität hinaus maskieren
            expert_mask = expert_mask * torch.lt(expert_rank, capacity)

            # Position im Expert Batch
            expert_rank = torch.sum(expert_mask * expert_rank, dim=-1)  # [K, num_tokens]

            # Wahrscheinlichkeiten mit Maske multiplizieren
            router_probs = router_probs.view(num_tokens, self.n_experts)[
                None, :
            ]  # [1, num_tokens, n_experts]
            expert_weights = expert_mask * router_probs  # [K, num_tokens, n_experts]

            # One-hot für Position in Expert Batch
            expert_rank_one_hot = F.one_hot(
                expert_rank, num_classes=capacity
            )  # [K, num_tokens, capacity]

            # Gewichte an Expert Batch Position
            expert_weights = torch.sum(
                expert_weights.unsqueeze(3) * expert_rank_one_hot.unsqueeze(2), dim=0
            )  # [num_tokens, n_experts, capacity]
            expert_mask = expert_weights.bool()

            # Expert Batches erstellen
            x_flat = x.view(num_tokens, d_model)
            expert_batches = (
                expert_mask.permute(1, 2, 0).type_as(x) @ x_flat
            )  # [n_experts, capacity, d_model]

        return expert_weights, expert_mask, expert_batches, router_logits

    def _compute_capacity(self, num_tokens: int) -> int:
        """Berechnet Expert Capacity"""
        capacity = math.floor(
            self.n_experts_active * self.capacity_factor * num_tokens / self.n_experts
        )
        capacity += capacity % 2  # Gerade Zahl für bessere Hardware-Nutzung
        return max(int(capacity), 2)  # Minimum 2 für kleine Batches


class ExpertMLP(nn.Module):
    """
    Batch von MLP Experten.
    Alle Experten haben die gleiche Architektur, aber unabhängige Gewichte.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        bias: bool = False,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.bias = bias

        # 4x hidden dimension (Standard für GPT)
        hidden_dim = 4 * d_model

        # Gewichte für alle Experten (batch matmul)
        self.w_fc = nn.Parameter(torch.empty(n_experts, d_model, hidden_dim))
        self.w_proj = nn.Parameter(torch.empty(n_experts, hidden_dim, d_model))

        if bias:
            self.fc_bias = nn.Parameter(torch.empty(n_experts, 1, hidden_dim))
            self.proj_bias = nn.Parameter(torch.empty(n_experts, 1, d_model))
        else:
            self.register_parameter("fc_bias", None)
            self.register_parameter("proj_bias", None)

        # Aktivierungsfunktion
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swiglu":
            # SwiGLU braucht extra Gewichte
            self.w_gate = nn.Parameter(torch.empty(n_experts, d_model, hidden_dim))
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unbekannte Aktivierung: {activation}")

        self.dropout = nn.Dropout(dropout)
        self.activation_type = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_experts, capacity, d_model]

        Returns:
            output: [n_experts, capacity, d_model]
        """
        # Erste Linear Layer mit batch matmul
        h = torch.bmm(x, self.w_fc)
        if self.bias:
            h = h + self.fc_bias

        # Aktivierung
        if self.activation_type == "swiglu":
            # SwiGLU: silu(x @ W_gate) * (x @ W_fc)
            gate = torch.bmm(x, self.w_gate)
            h = self.activation(gate) * h
        else:
            h = self.activation(h)

        # Zweite Linear Layer
        output = torch.bmm(h, self.w_proj)
        if self.bias:
            output = output + self.proj_bias

        output = self.dropout(output)

        return output


class MoELayer(nn.Module):
    """
    Vollständige Mixture-of-Experts Layer.
    Kombiniert Router und Experten.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        n_experts_active: int = 2,
        use_noisy_gating: bool = True,
        capacity_factor: float = 1.25,
        bias: bool = False,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.router = MoERouter(
            d_model=d_model,
            n_experts=n_experts,
            n_experts_active=n_experts_active,
            use_noisy_gating=use_noisy_gating,
            capacity_factor=capacity_factor,
        )

        self.experts = ExpertMLP(
            d_model=d_model,
            n_experts=n_experts,
            bias=bias,
            dropout=dropout,
            activation=activation,
        )

        self.n_experts = n_experts
        self.n_experts_active = n_experts_active

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
            load_balance_loss: Skalarer Load Balancing Loss
            router_z_loss: Skalarer Router Z-Loss
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Routing
        expert_weights, expert_mask, expert_batches, router_logits = self.router(x)

        # Expert Forward Pass
        expert_outputs = self.experts(expert_batches)  # [n_experts, capacity, d_model]

        # Outputs kombinieren (gewichteter Durchschnitt)
        expert_weights_flat = expert_weights.view(num_tokens, -1)  # [num_tokens, n_experts * capacity]
        expert_outputs_flat = expert_outputs.view(-1, d_model)  # [n_experts * capacity, d_model]
        output = expert_weights_flat @ expert_outputs_flat  # [num_tokens, d_model]
        output = output.view(batch_size, seq_len, d_model)

        # Auxiliary Losses berechnen
        load_balance_loss = self._compute_load_balance_loss(router_logits, expert_mask)
        router_z_loss = self._compute_router_z_loss(router_logits)

        return output, load_balance_loss, router_z_loss

    def _compute_load_balance_loss(
        self, router_logits: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Load Balancing Loss (Switch Transformer, Fedus et al. 2022)
        Encourages uniform distribution of tokens across experts.
        """
        batch_size, seq_len, n_experts = router_logits.shape
        num_tokens = batch_size * seq_len

        # Probability pro Expert
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, n_experts]
        prob_per_expert = torch.mean(router_probs, dim=(0, 1))  # [n_experts]

        # Token Ratio pro Expert
        with torch.no_grad():
            # expert_mask ist [num_tokens, n_experts, capacity]
            tokens_per_expert = torch.sum(expert_mask.float(), dim=(0, 2))  # [n_experts]
            tokens_per_expert = tokens_per_expert / (num_tokens * self.n_experts_active)

        # Dot product (scaled by n_experts)
        loss = self.n_experts * torch.sum(prob_per_expert * tokens_per_expert)

        return loss

    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Router Z-Loss (ST-MoE, Zoph et al. 2022)
        Penalisiert große Router Logits für numerische Stabilität.
        """
        # Squared logsumexp über Experten
        z_loss = torch.logsumexp(router_logits, dim=-1) ** 2.0  # [B, T]
        z_loss = torch.mean(z_loss)

        return z_loss
