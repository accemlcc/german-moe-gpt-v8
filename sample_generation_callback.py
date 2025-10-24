"""
Sample Generation Callback für MoE Training
Generiert Texte während des Trainings um Fortschritt zu beobachten
"""

import torch
from transformers import TrainerCallback, AutoTokenizer
from typing import Optional
import os


class SampleGenerationCallback(TrainerCallback):
    """
    Generiert Sample-Texte alle N Steps während des Trainings
    """

    def __init__(
        self,
        tokenizer,
        prompts: list[str],
        generate_every_n_steps: int = 100,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        output_dir: str = "./samples",
    ):
        """
        Args:
            tokenizer: HuggingFace Tokenizer
            prompts: Liste von Prompts für Generierung
            generate_every_n_steps: Generiere alle N Steps
            max_new_tokens: Max neue Tokens
            temperature: Sampling Temperature
            top_k: Top-k Sampling
            top_p: Nucleus Sampling
            output_dir: Ordner für Sample Outputs
        """
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generate_every_n_steps = generate_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.output_dir = output_dir

        # Output Ordner erstellen
        os.makedirs(output_dir, exist_ok=True)

        # Samples Log Datei
        self.log_file = os.path.join(output_dir, "generation_log.txt")

        # Header schreiben
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MoE Training - Sample Generation Log\n")
            f.write("=" * 80 + "\n\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Wird nach jedem Training Step aufgerufen
        """
        # Nur alle N Steps generieren
        if state.global_step % self.generate_every_n_steps != 0:
            return

        # Skip wenn kein Model
        if model is None:
            return

        print(f"\n{'='*80}")
        print(f"🎨 GENERATING SAMPLES @ STEP {state.global_step}")
        print(f"{'='*80}\n")

        # Model in Eval Mode
        model.eval()

        samples = []
        samples.append(f"\n{'='*80}\n")
        samples.append(f"Step: {state.global_step}\n")
        samples.append(f"{'='*80}\n\n")

        with torch.no_grad():
            for i, prompt in enumerate(self.prompts, 1):
                print(f"[{i}/{len(self.prompts)}] Prompt: '{prompt}'")

                # Tokenize
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                input_ids = input_ids.to(model.device)

                try:
                    # Generieren
                    # NOTE: repetition_penalty is REQUIRED for longer generations!
                    # For 300 tokens, 1.3-1.5 is better than 1.2
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        repetition_penalty=1.4,  # ← Higher for 300 tokens!
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode
                    generated_text = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )

                    # Ausgabe
                    print(f"   → {generated_text}\n")

                    # Log speichern
                    samples.append(f"Prompt {i}: {prompt}\n")
                    samples.append(f"Output: {generated_text}\n\n")

                except Exception as e:
                    error_msg = f"   ❌ Error: {str(e)}\n"
                    print(error_msg)
                    samples.append(f"Prompt {i}: {prompt}\n")
                    samples.append(f"Error: {str(e)}\n\n")

        # Samples in Datei schreiben
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.writelines(samples)

        print(f"{'='*80}\n")

        # Model zurück in Training Mode
        model.train()


def get_german_sample_prompts():
    """
    Gibt eine Liste deutscher Sample-Prompts zurück
    """
    return [
        "Die Künstliche Intelligenz",
        "Im finsteren Wald",
        "In der Zukunft werden wir",
        "Machine Learning bedeutet",
        "Das Wetter heute ist",
        "Ein wichtiger Aspekt der",
        "Die Geschichte von",
        "Wissenschaftler haben herausgefunden",
    ]
