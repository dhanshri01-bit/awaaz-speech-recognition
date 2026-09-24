"""
Step 3 (Hrs 16-26 prep): merge the LoRA adapter into the base model's
weights, then export to ONNX so it can run offline via onnxruntime
(desktop demo) or onnxruntime-mobile (real Android deployment later).

Usage:
    python scripts/export_onnx.py \
        --base_model openai/whisper-small \
        --adapter_dir adapters/speaker_01 \
        --out_dir adapters/speaker_01_onnx
"""
import argparse

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.exporters.onnx import main_export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openai/whisper-small")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    print("Merging LoRA adapter into base weights...")
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    merged = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = merged.merge_and_unload()

    merged_dir = args.out_dir + "_merged_hf"
    merged.save_pretrained(merged_dir)
    WhisperProcessor.from_pretrained(args.base_model).save_pretrained(merged_dir)

    print("Exporting to ONNX (this covers encoder + decoder + generation config)...")
    main_export(
        model_name_or_path=merged_dir,
        output=args.out_dir,
        task="automatic-speech-recognition-with-past",
    )
    print(f"Offline-ready model at {args.out_dir}")


if __name__ == "__main__":
    main()
