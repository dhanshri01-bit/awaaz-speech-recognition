"""
Step 4 (Hrs 16-26): the demo that carries the whole pitch — baseline
Whisper and the personalized adapter transcribing the SAME audio,
side by side, live. Corrections typed here get logged as new
training pairs for the next overnight fine-tune (Step 5 / Hrs 26-36).

Usage:
    streamlit run app/demo_app.py
"""
import csv
import io
import os
import time

import streamlit as st
import torch
import torchaudio
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

st.set_page_config(page_title="Awaaz — Live Demo", layout="wide")

BASE_MODEL = "openai/whisper-tiny"
ADAPTER_DIR = "adapters/speaker_01"
CORRECTIONS_LOG = "data/corrections.csv"


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    baseline_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
    baseline_model.eval()

    personal_model = None
    if os.path.isdir(ADAPTER_DIR):
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
        personal_model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
        personal_model.eval()

    return processor, baseline_model, personal_model, device


def transcribe(model, processor, waveform, device):
    inputs = processor(
        waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(inputs, max_new_tokens=200)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def log_correction(audio_path, personal_hyp, corrected_text):
    is_new = not os.path.exists(CORRECTIONS_LOG)
    with open(CORRECTIONS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["audio_path", "reference_text"])
        writer.writerow([audio_path, corrected_text])


st.title("Awaaz — Live Transcript Comparison")
st.caption(
    "This is the whole pitch in one screen: the same audio, "
    "transcribed by a generic model and by a model that knows this speaker."
)

processor, baseline_model, personal_model, device = load_models()

if personal_model is None:
    st.warning(
        f"No adapter found at `{ADAPTER_DIR}`. Run `scripts/finetune_lora.py` first — "
        "showing baseline only for now."
    )

audio_file = st.file_uploader("Upload or record a clip (wav)", type=["wav"])

if audio_file is not None:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)

    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Off-the-shelf Whisper")
        t0 = time.time()
        baseline_text = transcribe(baseline_model, processor, waveform, device)
        st.write(f"⏱ {time.time() - t0:.1f}s")
        st.error(baseline_text if baseline_text else "(no speech detected)")

    with col2:
        st.subheader("Awaaz (personalized)")
        if personal_model is not None:
            t0 = time.time()
            personal_text = transcribe(personal_model, processor, waveform, device)
            st.write(f"⏱ {time.time() - t0:.1f}s")
            st.success(personal_text if personal_text else "(no speech detected)")

            st.divider()
            st.caption("Wrong? Fix it — this becomes tonight's training data.")
            corrected = st.text_input("Correct transcript", value=personal_text)
            if st.button("Save correction"):
                tmp_path = f"data/clip_{int(time.time())}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                log_correction(tmp_path, personal_text, corrected)
                st.toast("Saved — will be used in the next fine-tune pass.")
        else:
            st.info("Adapter not loaded yet.")
else:
    st.info("Upload a WAV clip to compare transcripts.")