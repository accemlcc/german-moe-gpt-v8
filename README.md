# German MoE GPT v8 - OPUS EDITION Trainer

Dieses Repository enthält die Skripte und Konfigurationen zum Trainieren des deutschen 149.6M Parameter Mixture-of-Experts (MoE) Sprachmodells "German MoE GPT v8".

## Projekt-Status (Oktober 2025)

-   **v8 Pre-Training:** ✅ **COMPLETE**
-   **Fine-Tuning Phase:** 🔬 **IN PROGRESS**

## Übersicht

Das Ziel dieses Projekts war das Training eines leistungsfähigen, mittelgroßen deutschen Sprachmodells auf einer einzelnen Consumer-GPU (RTX 4090). Das v8-Basismodell wurde erfolgreich auf einem 17.4 GB großen, bereinigten Datensatz trainiert und zeigt eine hohe Kohärenz und quasi keinen SEO-Spam, was ein großer Fortschritt gegenüber früheren Versionen ist.

### Modell-Architektur

-   **Parameter:** 149.6M
-   **Architektur:** Mixture-of-Experts (MoE)
-   **Experten:** 32
-   **Aktive Parameter pro Token:** ~33%
-   **Kontextlänge:** 2048 Tokens
-   **Embedding:** Rotatory Position Embedding (RoPE)

### Trainings-Datensatz (v8 OPUS Mix)

-   **Clean German Wikipedia:** ~11 GB
-   **OpenSubtitles (German):** Dialog-Korpus
-   **Belletristik:** Erweiterter Korpus deutscher Literatur
-   **Gesamtgröße:** ~17.4 GB

### Pre-Training Ergebnisse

-   **Trainings-Fortschritt:** 300.000 / 300.000 Schritte
-   **Trainings-Verlust:** Reduziert von 12.0 → 2.55 (79% Reduktion)
-   **Evaluierungs-Verlust:** Reduziert von 4.58 → 2.40 (48% Reduktion)
-   **Finale Perplexität:** **11.0** (exp(2.40))
-   **Gesamttrainingszeit:** ca. 48 Stunden

## Benutzung

### 1. Umgebung einrichten

Stellen Sie sicher, dass alle Abhängigkeiten aus der `requirements.txt` installiert sind und aktivieren Sie die Conda-Umgebung:

```bash
conda activate nano_moe
```

### 2. Training starten / fortsetzen

Das Trainings-Skript findet automatisch den letzten verfügbaren Checkpoint im Verzeichnis `moe_checkpoints_v8_clean` und setzt das Training fort.

```bash
python train_moe_v8_clean.py
```

### 3. Inferenz / Textgenerierung

Um mit dem trainierten Modell Text zu generieren, kann das Inferenz-Skript verwendet werden. Es lädt standardmäßig den neuesten v8-Checkpoint.

```bash
python inference.py
```

### 4. Monitoring

Das Training kann mit TensorBoard und durch Beobachtung der Log-Dateien überwacht werden.

```bash
# TensorBoard starten
start_tensorboard.bat

# Generierte Samples live beobachten
tail -f samples_v8_clean/generation_log.txt

# GPU-Auslastung prüfen
nvidia-smi -l 1
```

## Wichtige Verzeichnisse

-   **Training-Skript:** `train_moe_v8_clean.py`
-   **Fine-Tuning Skript:** `finetune_mixed_qa_v2.py`
-   **Checkpoints:** `./moe_checkpoints_v8_clean/`
-   **Finale Modelle:** `./moe_final_v8_clean/`
-   **Logs:** `./logs_v8_clean/`
-   **Generierte Samples:** `./samples_v8_clean/`
