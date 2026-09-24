# What was actually verified, and what you still need to run

## The honest summary

I could not download real Whisper model weights in the sandbox I build files
in — Hugging Face, OpenAI's model CDN, and PyTorch's model hub are all
network-blocked there. So I could not produce real "before/after Whisper
word-error-rate" numbers for you here. Anything claiming to be a measured
accuracy number without this note would have been fabricated — I didn't
want to hand you that.

Here's what I did instead, all genuinely real:

## 1. Real synthetic audio data (`data/synthetic_demo/`)

20 short phrases, generated with `scripts/generate_data.py`:
- **`enroll/`** (15 clips) — for personalizing the adapter
- **`eval/`** (5 clips, different phrases) — held out, never trained on

Every clip is real audio: synthesized with `espeak-ng` (offline TTS), then
run through a **fixed, consistent distortion** (pitch/formant shift + tempo
change + slight tremolo via ffmpeg) applied identically to every clip. This
stands in for one person's atypical-but-consistent articulation pattern —
the same idea the pitch deck describes, applied synthetically so the
pipeline has something real to chew on. **This is disclosed synthetic data,
not a claim about real dysarthric speech** — say so if you use it in your
submission, and swap in real recordings the moment you have them.

## 2. A real, debugged pipeline

I ran your actual `finetune_lora.py` logic — the same LoRA config, the same
dataset loading, the same training step — against a same-architecture but
randomly-initialized Whisper model (since real weights weren't reachable),
using the real audio above. Result: **forward pass, backward pass, and
adapter save all worked correctly.** Along the way I found and fixed one
real bug: newer versions of the `datasets` library need `torchcodec` for
audio decoding, which wasn't in the original `requirements.txt`. It is now.

All four scripts (`baseline_wer.py`, `finetune_lora.py`, `export_onnx.py`,
`app/demo_app.py`) byte-compile cleanly with zero import errors against the
exact library versions in `requirements.txt`.

## 3. What you need to run, on your own machine or Google Colab

Anywhere `huggingface.co` isn't blocked (which is almost everywhere except
this particular sandbox):

```bash
pip install -r requirements.txt

# Real baseline — generic Whisper on the distorted eval set
python scripts/baseline_wer.py \
  --manifest data/synthetic_demo/eval_manifest.csv \
  --model openai/whisper-tiny

# Real personalization — LoRA adapter trained on the distorted enroll set
python scripts/finetune_lora.py \
  --manifest data/synthetic_demo/enroll_manifest.csv \
  --base_model openai/whisper-tiny \
  --out_dir adapters/speaker_01 \
  --epochs 8

# The live demo
streamlit run app/demo_app.py
```

This will download `openai/whisper-tiny` (~150MB) automatically on first
run. Total time on a free Colab T4 GPU: a few minutes end to end, given how
small this dataset is.

## What to say in your submission if you use this as-is

Be upfront: *"Speech data here is synthetically generated (TTS + a
consistent distortion profile) to validate the pipeline end-to-end, since
real dysarthric speech data wasn't accessible in the build window. The
architecture and personalization approach are the actual contribution; real
accuracy numbers require real recordings."* Judges respect disclosed scope
far more than numbers that don't hold up under a follow-up question.
