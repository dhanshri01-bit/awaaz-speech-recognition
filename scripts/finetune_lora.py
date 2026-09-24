"""
Step 2 (Hrs 6-16): fine-tune a LoRA adapter on ONE plot's own weekly
photos, on top of a frozen generic disease-classifier backbone.

Expects a manifest CSV with columns: audio_path,reference_text
This should be the same speaker's audio across several short clips.

Usage:
    python scripts/finetune_lora.py \
        --manifest data/synthetic_demo/enroll_manifest.csv \
        --base_model openai/whisper-tiny \
        --out_dir adapters/speaker_01 \
        --epochs 8
"""
import argparse
import csv

import soundfile as sf
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def load_dataset(manifest_path, processor):
    audio_paths, texts = [], []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            audio_paths.append(row["audio_path"])
            texts.append(row["reference_text"])

    ds = Dataset.from_dict({"audio_path": audio_paths, "text": texts})

    def prepare(batch):
        audio_array, sr = sf.read(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return ds.map(prepare, remove_columns=ds.column_names)


class DataCollator:
    """Pads input features and label ids separately, as Whisper needs."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_model", default="openai/whisper-small")
    ap.add_argument("--out_dir", default="adapters/speaker_01")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # Freeze everything, attach LoRA to the attention projections only.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be well under 1%

    dataset = load_dataset(args.manifest, processor)
    split = dataset.train_test_split(test_size=0.15, seed=42)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollator(processor),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # Save ONLY the adapter — this is the small personal file per user.
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()