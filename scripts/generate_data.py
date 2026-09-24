"""
Generates a small, entirely-offline synthetic speech dataset:
1. Clean reference audio via espeak-ng TTS for each phrase.
2. A DISTORTED version of each phrase using a fixed, consistent
   transformation (pitch shift + tempo irregularity + formant-ish
   EQ warp) applied identically every time — standing in for one
   person's consistent atypical articulation pattern, the way
   dysarthric speech is idiosyncratic-but-consistent per speaker.

This is disclosed synthetic data for pipeline validation, not a
claim about real dysarthric speech accuracy.
"""
import csv
import os
import subprocess

PHRASES = [
    "open whatsapp",
    "call my sister",
    "what is the time",
    "i need water",
    "send money to mummy",
    "turn on the light",
    "play some music",
    "read my messages",
    "how is the weather today",
    "set an alarm for morning",
    "video call my brother",
    "book a taxi",
    "check my bank balance",
    "remind me to take medicine",
    "close the door please",
    "increase the volume",
    "what is my schedule today",
    "send a message to doctor",
    "turn off the fan",
    "call for help",
]

ENROLL_PHRASES = PHRASES[:15]   # personalization set
EVAL_PHRASES = PHRASES[15:]     # held-out test set, never trained on

RAW_DIR = "raw"
ENROLL_DIR = "enroll"
EVAL_DIR = "eval"


def synthesize(text, out_path, speed=150, pitch=50):
    subprocess.run(
        ["espeak-ng", "-s", str(speed), "-p", str(pitch), "-v", "en", text, "-w", out_path],
        check=True, capture_output=True,
    )


def distort(in_path, out_path):
    """Fixed, consistent distortion profile applied to every clip —
    simulates one person's atypical, but internally consistent,
    articulation. Uses ffmpeg's atempo + asetrate (pitch/formant
    shift) + a slight tremolo to mimic articulatory instability.
    """
    filter_chain = (
        "asetrate=22050*0.88,"   # lower pitch/formants ~12%
        "atempo=1.18,"           # speed back up (net: pitch-shifted, timing off)
        "vibrato=f=4.5:d=0.35,"  # slight tremor -> articulatory instability
        "aresample=16000"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-af", filter_chain, "-ac", "1", out_path],
        check=True, capture_output=True,
    )


def resample_clean(in_path, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out_path],
        check=True, capture_output=True,
    )


def build_set(phrases, out_dir, manifest_path, apply_distortion):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, text in enumerate(phrases):
        raw_path = f"{RAW_DIR}/clip_{out_dir}_{i:02d}.wav"
        final_path = f"{out_dir}/clip_{i:02d}.wav"
        synthesize(text, raw_path)
        if apply_distortion:
            distort(raw_path, final_path)
        else:
            resample_clean(raw_path, final_path)
        rows.append((final_path, text))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "reference_text"])
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Building enrollment set (distorted — this speaker's voice profile)...")
    enroll_rows = build_set(ENROLL_PHRASES, ENROLL_DIR, "enroll_manifest.csv", apply_distortion=True)
    print(f"  {len(enroll_rows)} clips -> enroll_manifest.csv")

    print("Building held-out eval set (distorted, different phrases, never trained on)...")
    eval_rows = build_set(EVAL_PHRASES, EVAL_DIR, "eval_manifest.csv", apply_distortion=True)
    print(f"  {len(eval_rows)} clips -> eval_manifest.csv")

    print("Done.")
