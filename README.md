# sEMG-Based Subvocal Inner-Speech Decoding Device — Short Version

This project is an experimental system for decoding subvocal inner speech from surface electromyography (sEMG) signals produced by speech-related facial and submandibular muscles. It uses non-invasive electrodes, real-time signal processing, a deep-learning sEMG encoder, and a transformer language decoder to generate text from muscle activity. The goal is a future wearable communication interface for people with speech impairments, such as ALS. The project does **not** use EEG or decode thoughts directly from brain activity.

## System Flow

Surface EMG electrodes → ADS1299 analog front-end → ESP32-S3 → Wi-Fi/BLE → host computer → sEMG-to-text model → decoded text.

Key principles: non-invasive acquisition, low-latency streaming, modular hardware/software, reproducible experiments, and separable acquisition/inference components for privacy.

## Repository Components

- `recording_software/` — desktop data-collection and visualization interface.
- `ml/` — sEMG-to-text model, data loaders, training, evaluation, inference.
- `ml/data/` — preprocessed recordings, labels, metadata.
- `hardware/` — ESP32-S3 firmware for ADS1299 acquisition and streaming.
- `figures/` — architecture diagrams, schematics, signal visualizations, results.

## Data Collection and Processing

Recordings use custom sEMG hardware and software. The Harvard Sentences protocol includes 720 phonetically balanced sentences, 15 recordings each, for 10,800 recordings per complete pass. Metadata should document subject, session, electrode placement, and acquisition settings.

Processing includes filtering, notch filtering for mains interference, artifact handling, segmentation, and normalization. Preprocessing should be fitted on training data only and applied consistently to validation, test, and deployment data.

## Model

The decoding pipeline has three main stages:

1. **sEMG encoder** — learns temporal and cross-channel muscle patterns.
2. **Feature projection** — maps encoder output into the language model’s embedding space.
3. **Language decoder** — a transformer generates text conditioned on sEMG features.

Training should support configurable parameters, train/validation splits, checkpointing, early stopping, loss tracking, qualitative and quantitative evaluation. Train/test separation must prevent the same trial appearing in both sets; subject-independent evaluation is recommended.

## Hardware

The device uses surface EMG electrodes, an ADS1299 24-bit biopotential front-end (up to 8 channels), and an ESP32-S3 microcontroller. The ESP32-S3 reads ADS1299 data over SPI, buffers samples, frames multichannel data, and streams it over Wi-Fi or BLE. Wearable integration should account for isolation, safe charging, electrode contact, mechanical stability, and movement/wireless interference.

Firmware responsibilities: initialize hardware, configure channels/sampling, acquire and buffer samples, package frames, transmit data, and handle connection/buffer errors. Firmware and host must agree on channel order, sample rate, numeric format, and frame format.

## Quickstart

```bash
git clone https://github.com/alexsteinerr/emg-subvocal-decoding
cd emg-subvocal-decoding
python -m venv .venv
pip install -r requirements.txt
```

Collect data:

```bash
cd recording_software
python recording_software.py
```

Train:

```bash
python ml/scripts/train.py
```

Build/flash hardware:

```bash
cd hardware
pio run
pio run -t upload
```

Then start host-side inference, connect the device, verify frame format, and evaluate predictions against known target sentences before real-time use.

## Limitations

Subvocal sEMG decoding is experimental. Signals vary by participant, electrode placement, session, and intended utterance. Performance on speech-related muscle activity should not be assumed to demonstrate reliable decoding of purely imagined inner speech. Future work includes improving cross-session/participant generalization, reducing artifact sensitivity, low-latency streaming inference, separate evaluation of genuinely subvocalized tasks, wearable integration, and confidence estimation/user confirmation before executing decoded commands.