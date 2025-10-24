"""
German MoE GPT v8 - CLEAN DATA + OPUS EDITION
Training mit Wikipedia + OpenSubtitles + Belletristik

Datasets (v8 - CLEAN + DIALOGUES! 🎉):
- Clean Wikipedia (local) - 11 GB (64%)
- OpenSubtitles OPUS (local) - 4.2 GB (24%)
- Belletristik (arnomatic/merged_all) - 2.2 GB (12%)

Total: ~17.4 GB of 100% CLEAN German text!
NO spam, NO ads, NO SEO garbage! ✅
PLUS natural dialogues from movie subtitles! 🎬
"""

import os
import sys

# Disable HF transfer (can cause issues on Windows)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from datasets import load_dataset, interleave_datasets
from transformers import TrainingArguments, set_seed, AutoTokenizer

from moe_config import MoEGPTConfig
from moe_model import MoEGPTForCausalLM
from moe_trainer import MoETrainer, MoEEvalCallback, DataCollatorForLanguageModeling
from sample_generation_callback import SampleGenerationCallback, get_german_sample_prompts


def load_clean_datasets(tokenizer, max_length=2048, seed=42, resume_step=0):
    """
    Lädt 3 clean datasets (v8 - INTERLEAVED!):
    - Wikipedia (WITH EOS) - 64%
    - OpenSubtitles OPUS (NO EOS) - 24%
    - Belletristik (NO EOS) - 12%

    Args:
        resume_step: If > 0, adjusts seed to continue from checkpoint
    """
    # Adjust seed based on resume step (für reproducibility beim Resume)
    effective_seed = seed + (resume_step // 1000)
    print(f"📚 Lade CLEAN Datasets (v8 - OPUS Edition)...")
    if resume_step > 0:
        print(f"   🔄 Resume from step {resume_step} → Effective seed: {effective_seed}\n")
    else:
        print()

    # ========================================================================
    # 1. WIKIPEDIA (WITH EOS between articles)
    # ========================================================================
    print("1️⃣ Wikipedia (WITH EOS)...")
    try:
        wiki_ds = load_dataset(
            "jonas-is-coding/german-wikipedia-articles",
            split="train",
            streaming=True
        )
        print("   ✅ Dataset loaded (streaming mode)")

        # Shuffle
        print("   🔀 Shuffling with buffer_size=10,000...")
        wiki_ds = wiki_ds.shuffle(seed=effective_seed, buffer_size=10000)
        print("   ✅ Shuffle applied")

    except Exception as e:
        print(f"   ❌ Wikipedia Error: {e}")
        raise ValueError(f"Failed to load Wikipedia: {e}")

    # ========================================================================
    # 2. OPENSUBTITLES OPUS (NO EOS - continuous dialogues)
    # ========================================================================
    print("\n2️⃣ OpenSubtitles OPUS (NO EOS - continuous dialogues)...")
    try:
        opus_ds = load_dataset(
            "arnomatic/german-opus-subtitles",
            split="train",
            streaming=True
        )
        print("   ✅ Dataset loaded (streaming mode)")

        # Shuffle
        print("   🔀 Shuffling with buffer_size=10,000...")
        opus_ds = opus_ds.shuffle(seed=effective_seed, buffer_size=10000)
        print("   ✅ Shuffle applied")

    except Exception as e:
        print(f"   ❌ OpenSubtitles Error: {e}")
        raise ValueError(f"Failed to load OpenSubtitles: {e}")

    # ========================================================================
    # 3. BELLETRISTIK (NO EOS - continuous)
    # ========================================================================
    print("\n3️⃣ Belletristik (NO EOS - continuous)...")
    try:
        belle_ds = load_dataset(
            "arnomatic/merged_all",
            split="train",
            streaming=True
        )
        print("   ✅ Dataset loaded (streaming mode)")

        # Shuffle
        print("   🔀 Shuffling with buffer_size=10,000...")
        belle_ds = belle_ds.shuffle(seed=effective_seed, buffer_size=10000)
        print("   ✅ Shuffle applied")

    except Exception as e:
        print(f"   ❌ Belletristik Error: {e}")
        raise ValueError(f"Failed to load Belletristik: {e}")

    print("\n✅ All datasets loaded!")
    print("   Wikipedia: 4 GB (WITH EOS)")
    print("   OpenSubtitles: 4.2 GB (NO EOS)")
    print("   Belletristik: 2.2 GB (NO EOS)")
    print("   Total: ~10.4 GB clean German!")

    # ========================================================================
    # DIRECT PACKING (no intermediate tokenization)
    # ========================================================================
    print("\n🔤 Tokenizing & Packing datasets...")

    from datasets import IterableDataset as HFIterableDataset

    def pack_dataset_with_eos(dataset, text_field='text'):
        """Pack dataset WITH EOS directly into 2048-token batches"""
        def gen():
            buffer = []
            for example in dataset:
                text = example.get(text_field, '')
                if not text or not text.strip():
                    continue

                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Add tokens + EOS
                buffer.extend(tokens)
                buffer.append(tokenizer.eos_token_id)

                # Yield complete chunks
                while len(buffer) >= max_length:
                    yield {
                        "input_ids": buffer[:max_length],
                        "attention_mask": [1] * max_length,
                        "labels": buffer[:max_length],
                    }
                    buffer = buffer[max_length:]

        return HFIterableDataset.from_generator(gen)

    def pack_dataset_no_eos(dataset, text_field='text'):
        """Pack dataset WITHOUT EOS directly into 2048-token batches"""
        def gen():
            buffer = []
            for example in dataset:
                text = example.get(text_field, '')
                if not text or not text.strip():
                    continue

                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Add tokens (NO EOS)
                buffer.extend(tokens)

                # Yield complete chunks
                while len(buffer) >= max_length:
                    yield {
                        "input_ids": buffer[:max_length],
                        "attention_mask": [1] * max_length,
                        "labels": buffer[:max_length],
                    }
                    buffer = buffer[max_length:]

        return HFIterableDataset.from_generator(gen)

    print("   Wikipedia (WITH EOS)...")
    wiki_batched = pack_dataset_with_eos(wiki_ds, text_field='content')

    print("   OpenSubtitles (NO EOS)...")
    opus_batched = pack_dataset_no_eos(opus_ds, text_field='text')

    print("   Belletristik (NO EOS)...")
    belle_batched = pack_dataset_no_eos(belle_ds, text_field='text')

    print("✅ Batching complete!")

    # ========================================================================
    # INTERLEAVE DATASETS (64% Wiki, 24% OPUS, 12% Belle)
    # ========================================================================
    print("\n🔀 Interleaving datasets (64/24/12)...")

    train_dataset = interleave_datasets(
        [wiki_batched, opus_batched, belle_batched],
        probabilities=[0.64, 0.24, 0.12],
        seed=effective_seed,
        stopping_strategy="all_exhausted"
    )

    print("✅ Datasets interleaved! (v8 strategy)")
    print("   Wikipedia: 64%")
    print("   OpenSubtitles: 24%")
    print("   Belletristik: 12%")

    # ========================================================================
    # EVAL DATASET (fixed 500 samples from Wikipedia)
    # ========================================================================
    eval_dataset_path = "./eval_dataset_v8_clean"

    if os.path.exists(eval_dataset_path):
        print(f"\n📊 Loading existing eval dataset from {eval_dataset_path}...")
        from datasets import load_from_disk
        eval_dataset = load_from_disk(eval_dataset_path)
        print(f"✅ Eval dataset loaded: {len(eval_dataset)} samples (from disk)")
    else:
        print("\n📊 Creating fixed eval set (500 samples from Wikipedia)...")

        eval_samples = []
        eval_iter = iter(wiki_batched)
        for i in range(500):
            try:
                sample = next(eval_iter)
                eval_samples.append(sample)
                if (i + 1) % 100 == 0:
                    print(f"   Collected {i+1}/500 samples...")
            except StopIteration:
                print(f"   ⚠️  Only {i} eval samples available (dataset exhausted)")
                break

        if len(eval_samples) == 0:
            raise ValueError("No eval samples collected! Dataset exhausted immediately.")

        print(f"   Collected {len(eval_samples)} samples total")

        # Convert to regular Dataset (not streaming!)
        from datasets import Dataset
        eval_dataset = Dataset.from_dict({
            key: [sample[key] for sample in eval_samples]
            for key in eval_samples[0].keys()
        })

        # Save to disk
        print(f"💾 Saving eval dataset to {eval_dataset_path}...")
        eval_dataset.save_to_disk(eval_dataset_path)
        print(f"✅ Eval dataset saved to disk!")

    print(f"   → No more fsspec cache leak!")
    print(f"   Training: Clean Mix (streaming)")
    print(f"   Eval: {len(eval_dataset)} samples (fixed, from disk)\n")

    return train_dataset, eval_dataset


def main():
    SEED = 42
    set_seed(SEED)

    # Config
    config = MoEGPTConfig(
        vocab_size=128256,
        n_positions=2048,
        n_embd=512,
        n_layer=8,
        n_head=8,
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
        rope_theta=10000.0,
    )

    print("\n🔧 Model Config:")
    print(f"   - Experten: {config.n_experts} (Top-{config.n_experts_active})")
    print(f"   - Parameter: {config.total_experts} MoE experts")

    # Training Args
    # Dataset: ~10.4 GB ≈ 2.5B tokens ≈ 1.2M batches (2048 tokens each)
    # With batch size 32: ~38K steps per epoch
    # ~1.3 epochs = ~50K steps (interleaved = more efficient)
    training_args = TrainingArguments(
        output_dir="./moe_checkpoints_v8_clean",
        run_name="german_moe_v8_clean",
        max_steps=200000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=6e-4,
        warmup_steps=2000,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_dir="./logs_v8_clean",
        logging_steps=100,
        logging_first_step=True,
        report_to=["tensorboard"],
        eval_strategy="steps",
        eval_steps=1000,  # Every 1K steps (more frequent than v7)
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        seed=SEED,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ignore_data_skip=True,  # CRITICAL: Don't skip batches, use fresh shuffled data!
    )

    # Check for existing checkpoints (auto-resume) - DO THIS EARLY!
    import glob
    checkpoints = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
    resume_from_checkpoint = None
    resume_step = 0

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        resume_from_checkpoint = latest_checkpoint
        resume_step = int(latest_checkpoint.split("-")[-1])
        print(f"\n🔄 RESUME Training from: {latest_checkpoint} (Step {resume_step})")
    else:
        print("\n🆕 Starting fresh training (no checkpoints found)")

    # Tokenizer
    print("\n📚 Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Llama 3.2 Tokenizer geladen")

    # Load Clean Datasets (with resume_step for reproducibility!)
    train_dataset, eval_dataset = load_clean_datasets(
        tokenizer=tokenizer,
        max_length=2048,
        seed=SEED,
        resume_step=resume_step,
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)

    # Model
    print("\n🏗️  Erstelle MoE Modell...")
    model = MoEGPTForCausalLM(config)

    # Ensure weight tying (especially after checkpoint load)
    model.tie_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modell erstellt! ({total_params/1e6:.1f}M params)")

    # Callbacks
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        prompts=get_german_sample_prompts(),
        generate_every_n_steps=1000,  # Every 1K steps - fast feedback!
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.7,
        output_dir="./samples_v8_clean",
    )

    # Trainer
    print("\n🚀 Initialisiere Trainer...")
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MoEEvalCallback(), sample_callback],
    )

    print("✅ Trainer bereit!")

    print("\n" + "=" * 60)
    print("🎯 STARTE TRAINING v8 - OPUS EDITION!")
    print("=" * 60)
    print("\nDataset Composition (INTERLEAVED!):")
    print("  Wikipedia (WITH EOS): 64%")
    print("  OpenSubtitles OPUS (NO EOS): 24%")
    print("  Belletristik (NO EOS): 12%")
    print("\nTotal: ~10.4 GB CLEAN German!")
    print("NO spam, NO ads, NO SEO garbage! 🎉")
    print("PLUS natural dialogues from movie subtitles! 🎬")
    print("=" * 60 + "\n")

    # Train with resume support
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save
    print("\n💾 Speichere finales Modell...")
    final_model_path = "./moe_final_v8_clean"
    trainer.save_model(final_model_path)
    config.save_pretrained(final_model_path)
    print(f"✅ Modell gespeichert in: {final_model_path}")

    # Eval
    print("\n📊 Finale Evaluation...")
    eval_results = trainer.evaluate()

    for key, value in eval_results.items():
        print(f"   - {key}: {value:.4f}")

    if "eval_loss" in eval_results:
        perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
        print(f"\n🎯 Finale Perplexity: {perplexity:.2f}")

    print("\n" + "=" * 60)
    print("✅ TRAINING ABGESCHLOSSEN!")
    print("=" * 60)


if __name__ == "__main__":
    main()
