"""
Step 1 (Hrs 0-6): Measure how badly off-the-shelf Whisper does on
dysarthric / atypical speech. This number is your problem statement.

Expects a manifest CSV with columns: audio_path,reference_text
Build one from TORGO (https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html),
UASpeech, or your own recordings of the target speaker.

Usage:
    python scripts/baseline_wer.py --manifest data/eval_manifest.csv --model openai/whisper-small
"""
import argparse
import csv

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio


def load_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["audio_path"], row["reference_text"]))
    return rows


def transcribe(audio_path, processor, model, device):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(
        waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        ids = model.generate(inputs, max_new_tokens=256)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", default="data/baseline_predictions.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    rows = load_manifest(args.manifest)
    refs, hyps = [], []

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "reference", "hypothesis"])
        for audio_path, ref in rows:
            hyp = transcribe(audio_path, processor, model, device)
            refs.append(ref)
            hyps.append(hyp)
            writer.writerow([audio_path, ref, hyp])
            print(f"REF: {ref}\nHYP: {hyp}\n")

    wer = jiwer.wer(refs, hyps)
    print(f"\n=== Baseline WER ({args.model}): {wer * 100:.1f}% over {len(rows)} utterances ===")
    print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
