"""
HuggingFace-compatible MoE Configuration
Basierend auf dem nanoMoE Blog Post
"""

from transformers import PretrainedConfig


class MoEGPTConfig(PretrainedConfig):
    """
    Konfiguration für MoE-basiertes GPT Modell.

    Args:
        vocab_size (int): Größe des Vokabulars
        n_positions (int): Maximale Sequenzlänge
        n_embd (int): Dimensionalität der Embeddings (d im Blog)
        n_layer (int): Anzahl der Transformer Blocks
        n_head (int): Anzahl der Attention Heads
        n_experts (int): Anzahl der Experten pro MoE Layer
        n_experts_active (int): Anzahl aktiver Experten (top-k)
        moe_layer_frequency (int): Jede n-te Layer wird zu MoE (P im Blog)
        capacity_factor (float): Expert Capacity Factor für Training
        eval_capacity_factor (float): Expert Capacity Factor für Evaluation
        use_noisy_gating (bool): Ob Noisy Top-k Gating verwendet werden soll
        aux_loss_alpha (float): Skalierung für Load Balancing Loss
        router_z_loss_alpha (float): Skalierung für Router Z-Loss
        bias (bool): Ob Bias in Linear Layers verwendet werden soll
        dropout (float): Dropout Probability
        activation_function (str): Aktivierungsfunktion (gelu, relu, swiglu)
        initializer_range (float): Standard Deviation für Weight Initialization
        layer_norm_epsilon (float): Epsilon für Layer Normalization
    """

    model_type = "moe_gpt"

    def __init__(
        self,
        vocab_size=128256,  # Llama 3.2 tokenizer (inkl. special tokens)
        n_positions=2048,  # Default 2048 für RoPE
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_experts=8,
        n_experts_active=2,
        moe_layer_frequency=2,
        capacity_factor=1.25,
        eval_capacity_factor=2.0,
        use_noisy_gating=True,
        aux_loss_alpha=0.01,
        router_z_loss_alpha=0.001,
        bias=False,
        dropout=0.1,
        activation_function="gelu",
        initializer_range=0.1,
        layer_norm_epsilon=1e-5,
        use_cache=True,
        rope_theta=10000.0,  # RoPE base theta
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_experts = n_experts
        self.n_experts_active = n_experts_active
        self.moe_layer_frequency = moe_layer_frequency
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.use_noisy_gating = use_noisy_gating
        self.aux_loss_alpha = aux_loss_alpha
        self.router_z_loss_alpha = router_z_loss_alpha
        self.bias = bias
        self.dropout = dropout
        self.activation_function = activation_function
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        # HuggingFace Standard Attribute (für .generate())
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.max_position_embeddings = n_positions

        # Validierung
        assert n_embd % n_head == 0, "n_embd muss durch n_head teilbar sein"
        assert n_experts_active <= n_experts, "n_experts_active darf nicht größer als n_experts sein"
        assert moe_layer_frequency >= 1, "moe_layer_frequency muss mindestens 1 sein"

    @property
    def head_dim(self):
        """Dimension pro Attention Head"""
        return self.n_embd // self.n_head

    @property
    def total_experts(self):
        """Gesamtanzahl der Experten im Modell"""
        num_moe_layers = sum(1 for i in range(self.n_layer) if i % self.moe_layer_frequency == 0)
        return num_moe_layers * self.n_experts

    @property
    def active_parameters_ratio(self):
        """Ratio der aktiven Parameter (ungefähr)"""
        num_moe_layers = sum(1 for i in range(self.n_layer) if i % self.moe_layer_frequency == 0)
        num_dense_layers = self.n_layer - num_moe_layers

        # Vereinfachte Schätzung (ignoriert Attention)
        dense_params = num_dense_layers * (8 * self.n_embd**2)  # FFN params
        moe_total_params = num_moe_layers * self.n_experts * (8 * self.n_embd**2)
        moe_active_params = num_moe_layers * self.n_experts_active * (8 * self.n_embd**2)

        total = dense_params + moe_total_params
        active = dense_params + moe_active_params

        return active / total if total > 0 else 1.0
