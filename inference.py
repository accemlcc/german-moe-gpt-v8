"""
Inference Script für trainiertes MoE Modell
Lädt automatisch den neuesten Checkpoint und testet verschiedene Sampling Strategien
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from moe_config import MoEGPTConfig
from moe_model import MoEGPTForCausalLM

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def find_latest_checkpoint(checkpoint_dir="./moe_checkpoints_v8_clean"):
    """
    Findet den neuesten Checkpoint automatisch (v8 OPUS Edition!)

    Returns:
        str: Pfad zum neuesten Checkpoint oder None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Neuesten Checkpoint finden (nach creation time)
    latest = max(checkpoints, key=os.path.getctime)

    # Step Number extrahieren
    step = latest.split("checkpoint-")[-1]
    print(f"\n🔍 Neuester Checkpoint gefunden: Step {step}")

    return latest


def load_model(model_path=None, device="cuda"):
    """
    Lädt trainiertes MoE Modell
    Wenn model_path=None, wird automatisch der neueste Checkpoint geladen

    Args:
        model_path: Pfad zum gespeicherten Modell (None = auto-find)
        device: Device für Inference (cuda/cpu)

    Returns:
        model: Geladenes Modell
        config: Model Config
    """
    # Auto-find neuesten Checkpoint
    if model_path is None:
        model_path = find_latest_checkpoint()
        if model_path is None:
            # Fallback: Versuche finales Modell (v8)
            model_path = "./moe_final_v8_clean"
            if not os.path.exists(model_path):
                raise ValueError("Kein Checkpoint gefunden! Trainiere zuerst ein Modell.")

    print(f"\n📥 Lade Modell von: {model_path}")

    config = MoEGPTConfig.from_pretrained(model_path)
    model = MoEGPTForCausalLM.from_pretrained(model_path)

    # Auf Device verschieben
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Modell geladen auf GPU")
    else:
        model = model.cpu()
        print(f"✅ Modell geladen auf CPU")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   📊 Parameter: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   🧠 Experten: {config.total_experts}")
    print(f"   ⚡ Aktive Params: {config.active_parameters_ratio:.1%}")

    return model, config


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=400,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.0,
    device="cuda",
):
    """
    Generiert Text mit dem MoE Modell

    Args:
        model: MoE Modell
        tokenizer: Tokenizer
        prompt: Input Prompt (String)
        max_new_tokens: Maximale neue Tokens (400!)
        temperature: Sampling Temperature
        top_k: Top-k Sampling
        top_p: Nucleus Sampling
        repetition_penalty: Penalty für Wiederholungen
        device: Device

    Returns:
        generated_text: Generierter Text
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    if device == "cuda":
        input_ids = input_ids.cuda()

    # Generieren
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def test_sampling_strategies(model, tokenizer, prompts, device="cuda"):
    """
    Testet verschiedene Sampling Strategien

    Args:
        model: MoE Modell
        tokenizer: Tokenizer
        prompts: Liste von Test-Prompts
        device: Device
    """
    # Optimale Strategien (basierend auf umfangreichen Tests)
    strategies = {
        "Standard (temp=0.7, rep=1.2, top_k=50, top_p=0.8)": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
        },
        "Focused (temp=0.7, rep=1.4, #top_k=30, top_p=0.7)": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.7,  
            "repetition_penalty": 1.4,
        },
    }

    print("\n" + "=" * 80)
    print("🧪 TESTING SAMPLING STRATEGIES")
    print("=" * 80)

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: '{prompt}'")
        print(f"{'='*80}\n")

        for strategy_name, params in strategies.items():
            print(f"\n🎯 Strategy: {strategy_name}")
            print("-" * 80)

            try:
                generated = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=400,  # 400 Tokens!
                    device=device,
                    **params
                )

                print(f"{generated}")
                print()

            except Exception as e:
                print(f"❌ Error: {str(e)}\n")

    print("\n" + "=" * 80)
    print("💡 EMPFEHLUNG")
    print("=" * 80)
    print("""

    """)


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Modell laden (automatisch neuester Checkpoint!)
    model, config = load_model(model_path=None, device=device)

    # Tokenizer laden
    print("\n📚 Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Llama 3.2 Tokenizer geladen")
    print(f"   - Vocab Size: {tokenizer.vocab_size:,}")
    print(f"   - EOS Token: {tokenizer.eos_token}")

    # ==================== SAMPLING STRATEGY TESTS ====================

    # Test Prompts (diverse!)
    test_prompts = [
      "Gestern bin ich ",  # Narrativ
      "Der Mond ",  # Poetisch
      "Im Labor ",  # Wissenschaftlich
      "Hast du auch das Gefühl, dass",  # Persönlich/Forum
      "Die Zeit",
      "Was ist die Definition von Philosophie?"
  ]

    # Teste verschiedene Sampling Strategien
    test_sampling_strategies(model, tokenizer, test_prompts, device)


if __name__ == "__main__":
    main()
