"""
Custom MoE Trainer mit erweiterten Logging-Funktionen
"""

import torch
from typing import Dict, Optional, Any
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


class MoETrainer(Trainer):
    """
    Erweiterter Trainer für MoE Modelle mit speziellem Logging für:
    - Auxiliary Losses (Load Balancing, Router Z-Loss)
    - Expert Utilization
    - Capacity Factor Anpassung
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Überschreibt compute_loss um MoE-spezifische Losses zu berücksichtigen.
        Diese sind bereits im model.forward() eingerechnet, aber wir loggen sie separat.
        """
        # Labels für next token prediction
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()

        # Forward pass
        outputs = model(**inputs)

        # Loss ist bereits total loss (LM + aux losses)
        loss = outputs.loss

        # Logging der Auxiliary Losses (wenn im Training)
        if self.state.global_step % self.args.logging_steps == 0:
            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
                self.log({"train/aux_loss": outputs.aux_loss.item()})

            if hasattr(outputs, "router_z_loss") and outputs.router_z_loss is not None:
                self.log({"train/router_z_loss": outputs.router_z_loss.item()})

            # Gesamter Loss breakdown
            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
                lm_loss = (
                    loss.item()
                    - self.model.config.aux_loss_alpha * outputs.aux_loss.item()
                    - self.model.config.router_z_loss_alpha * outputs.router_z_loss.item()
                )
                self.log({"train/lm_loss": lm_loss})

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Überschreibt prediction_step um eval_loss korrekt zurückzugeben
        """
        # Labels sicherstellen
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()

        # Standard prediction_step aufrufen
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        return loss, logits, labels

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        Erweitert das Standard-Logging um MoE-spezifische Metriken
        """
        # GPU Memory Tracking
        if torch.cuda.is_available():
            logs["gpu_memory_allocated_gb"] = (
                torch.cuda.memory_allocated() / 1024**3
            )
            logs["gpu_memory_reserved_gb"] = (
                torch.cuda.memory_reserved() / 1024**3
            )

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


class MoEEvalCallback(TrainerCallback):
    """
    Callback für erweiterte MoE-spezifische Evaluation
    """

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """
        Nach jeder Evaluation loggen wir zusätzliche MoE Metriken
        """
        if metrics is not None and model is not None:
            # Model Statistiken
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            metrics["model/total_params_M"] = total_params / 1e6
            metrics["model/trainable_params_M"] = trainable_params / 1e6

            # MoE Spezifisch
            if hasattr(model.config, "n_experts"):
                metrics["model/total_experts"] = model.config.total_experts
                metrics["model/active_params_ratio"] = (
                    model.config.active_parameters_ratio
                )


class DataCollatorForLanguageModeling:
    """
    Einfacher Data Collator für Causal Language Modeling.
    Geht davon aus, dass Daten bereits tokenisiert sind.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        """
        Args:
            examples: Liste von Dicts mit 'input_ids' und 'attention_mask'

        Returns:
            Batch dict mit gepaddetem input_ids und attention_mask
        """
        # Maximale Länge in diesem Batch
        max_length = max(len(ex["input_ids"]) for ex in examples)

        input_ids = []
        attention_mask = []

        for ex in examples:
            seq_len = len(ex["input_ids"])
            padding_length = max_length - seq_len

            # Padding rechts
            padded_input_ids = ex["input_ids"] + [self.pad_token_id] * padding_length
            padded_attention_mask = ex["attention_mask"] + [0] * padding_length

            input_ids.append(padded_input_ids)
            attention_mask.append(padded_attention_mask)

        # Als Tensoren
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        return batch


def compute_metrics(eval_preds):
    """
    Compute Perplexity für Evaluation
    """
    predictions, labels = eval_preds

    # Für Language Modeling sind predictions die Logits
    # Labels sind die tatsächlichen Token IDs
    # Wir berechnen nur Perplexity hier (Loss wird automatisch geloggt)

    # Diese Funktion ist optional - Loss wird bereits vom Trainer berechnet
    return {}
