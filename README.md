# Awaaz: personalized speech recognition for atypical speech

**Live demo:** https://awaaz-speech-recognition-mrjq3appuqf4wvwj4tmb9tg.streamlit.app/
Awaaz trains a small personal LoRA adapter on top of OpenAI's Whisper, so it can learn one person's speech pattern instead of averaging across everyone. The demo transcribes the same clip twice, once with generic Whisper and once with the personalized adapter, so you can compare them side by side.

**Note on data:** the demo uses synthetic speech (text-to-speech audio with a consistent distortion applied), because real dysarthric speech recordings weren't available during the hackathon. The pipeline itself is real: the model, the training, the adapter and the live inference all work. Validating on real speakers is the next step.

Built for Global Innovation Hackathon 2026 (Bharat Academix), Accessibility & Inclusion track, by Dhanshri Wagh.

---


# Awaaz — build order

Matches the hackathon timeline. Run these in sequence; each stage produces
something you can show even if you run out of time before the next one.

## 0. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Get data (Hrs 0-6)

You need two small CSVs shaped like `data/sample_manifest.csv`
(`audio_path,reference_text`, one row per utterance, 16kHz mono WAV is safest):

- **`data/eval_manifest.csv`** — a held-out test set the model never trains on.
  Fastest path: pull a handful of speakers from the [TORGO dysarthric speech
  database](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) or
  [UASpeech](http://www.isle.illinois.edu/sst/data/UASpeech/). Use one speaker
  as your eval set.
- **`data/enroll_manifest.csv`** — ~15-20 minutes of ONE speaker's recordings
  for personalization. If you don't have dysarthric speech access in time,
  simulate it: apply pitch/speed/formant distortion to your own recordings
  (torchaudio has all three) — judges care that the *method* generalizes,
  not that your demo speaker has a real diagnosis.

## 2. Baseline WER (Hrs 0-6)

```bash
python scripts/baseline_wer.py --manifest data/eval_manifest.csv
```

This prints the number your whole pitch is built around. Write it down —
it goes on the problem-statement slide.

## 3. Personalize (Hrs 6-16)

```bash
python scripts/finetune_lora.py \
  --manifest data/enroll_manifest.csv \
  --out_dir adapters/speaker_01 \
  --epochs 8
```

Re-run `baseline_wer.py` logic against this adapter (swap in `PeftModel.from_pretrained`,
see `demo_app.py` for the pattern) to get your "after" number on the *same*
held-out set.

## 4. Demo (Hrs 16-26)

```bash
streamlit run app/demo_app.py
```

Upload a held-out clip. Baseline and personalized transcripts render side by
side. This screen is the pitch — rehearse it more than the slides.

## 5. Close the loop (Hrs 26-36)

Every "Save correction" click in the demo appends to `data/corrections.csv`.
Merge that into `enroll_manifest.csv` and re-run step 3 to show the adapter
improving live, or as a pre-recorded before/after if live retraining is too
slow to demo in real time.

## 6. Offline export (stretch, Hrs 36-48)

```bash
python scripts/export_onnx.py \
  --adapter_dir adapters/speaker_01 \
  --out_dir adapters/speaker_01_onnx
```

Only needed if you want to prove "runs on a phone, not the cloud" with an
actual offline artifact rather than an architecture diagram.

## Honesty checklist before you present

- [ ] The WER numbers you show are ones you actually measured on held-out audio
- [ ] You say "target" out loud for anything not yet measured
- [ ] You disclose if the enrolment speaker's speech was simulated, not real dysarthria
