# SPHiTZ - Aholab/HiTZ Speech Processing Pipeline (Diarization + ASR + C&P)

This repository provides a full end-to-end pipeline for **speaker diarization**, **speech-to-text transcription**, **optional translation / correction & paraphrasing (C&P)**, and **subtitle generation**.

It integrates:

- **Pyannote** for Voice Activity Detection (VAD) or Speaker Diarization  
- **NVIDIA NeMo** for ASR with word-level timestamps  
- **MarianMT** (HuggingFace) for translation / C&P  
- Multiple output formats: `.rttm`, `.xml`, `.json`, `.txt`, `.vtt`, `.srt`, `.png` waveform overlays

---

## Modes

| Mode  | Description | Outputs |
|-------|-------------|----------|
| `diar` | Only diarization | `.rttm`, `.png` |
| `all` | Diarization + ASR + segmentation + C&P + subtitles | `.xml`, `.rttm`, `.json`, `.txt`, `.srt`, `.vtt`, and optionally `_cp.*` |
| `cp` | Only text C&P via MarianMT | text result |

---

## Features

- **Speaker diarization** using Pyannote with configurable YAML pipeline.  
- **ASR with timestamps**, confidence, word-level metadata.  
- **Smart segmentation**: split long utterances by duration + character thresholds.  
- **GPU‑safe inference** with recursive fallback on OOM errors.  
- **Subtitle‑ready post‑processing**: padding + overlap avoidance.  
- **Multiple formats**: XML, RTTM, JSONL, TXT, SRT, VTT.  
- **Plotting**: waveform + speaker‑bar visualization.  
- **Compression**: automatically packages results into ZIP.

---

## Installation

### 1. Create environment

```bash
conda create -n audio-pipeline python=3.10
conda activate audio-pipeline
```

### 2. Install required libraries

Install PyTorch according to your CUDA setup:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Install NeMo:

```bash
pip install nemo_toolkit[asr]
```

Install pyannote.audio:

```bash
pip install pyannote-audio
```

Install transformers:

```bash
pip install transformers
```

Other Python packages used: `yaml`, `json`, `shutil`, `time`, `os`, `xml`, `matplotlib`, `tqdm`, `omegaconf`.

> **GPU strongly recommended** for Pyannote ≥3.x and NeMo ASR.

---

## Project Structure (functions)

- `pyannote_seg()` — diarization or VAD  
- `nemo_asr()` / `nemo_inference()` — ASR with timestamps  
- `marianmt_cp()` — capitalization/punctuation  
- `divide_segments()` — split long utterances  
- `add_padding()` — avoid subtitle overlaps  
- `map_speaker_color()` — assign colors  
- Writers: `to_rttm()`, `to_xml()`, `to_json()`, `to_vtt()`, `to_srt()`, `to_txt()`  
- Runners: `run_diar()`, `run_cp()`, `run_all()`

---

## Input Requirements

### Audio
- Formats supported by `torchaudio`: `.wav`, `.mp3`, `.flac`  
- Auto‑converted to **mono 16 kHz** (required by NeMo)

### Text (`run_mode="cp"`)
Plain text string or file.

---

## Output Files

| Format | Description |
|--------|-------------|
| `.png` | Diarization visualization |
| `.rttm` | Standard diarization file |
| `.xml` | Segment + word timing export |
| `.json` | JSONL with metadata |
| `.txt` | Transcript |
| `_cp.txt` | C&P / translated text |
| `.srt` | Subtitles |
| `_cp.srt` | C&P subtitles |
| `.vtt` | WebVTT |
| `_cp.vtt` | C&P WebVTT |
| `.zip` | Packaged output |

---

## Command‑Line Usage

```bash
python script.py --run_mode <mode> [arguments...]
```

### Arguments

| Flag | Description |
|------|-------------|
| `--run_mode` | `all`, `diar`, or `cp` |
| `--audio_file` | Audio input |
| `--input_text` | Text input (C&P mode) |
| `--out_path` | Output directory |
| `--seg_model` | Pyannote checkpoint |
| `--seg_config_yml` | Pyannote pipeline YAML |
| `--seg_option` | `diar` or `vad` |
| `--stt_model` | NeMo ASR checkpoint |
| `--cp_model` | MarianMT model |
| `--device` | `cuda` or `cpu` |

---

## Examples

### 1. Diarization only

```bash
python script.py     --run_mode diar     --audio_file input.wav     --seg_model /models/pyannote/model.ckpt     --seg_config_yml config.yaml     --out_path results     --device cuda
```

### 2. Full pipeline

```bash
python script.py     --run_mode all     --audio_file input.wav     --seg_model /models/pyannote/model.ckpt     --seg_config_yml config.yaml     --seg_option diar     --stt_model /models/nemo/stt_es.nemo     --cp_model /models/mt/eu_norm-eu     --out_path results     --device cuda
```

### 3. C&P only

```bash
python script.py     --run_mode cp     --input_text "kaixo mundua"     --cp_model /models/mt/eu_norm-eu     --device cuda
```

---

## Language Handling

If the ASR model filename contains:

- `"eu"` → Basque  
- `"es"` → Spanish  

---

## Output Organization

Results stored in:

```
<out_path>/<audio_filename>/
```

Generated ZIP:

```
result_<timestamp>.zip
```

---

## Visualization

`plot_diarization_sample()` outputs:

```
audio.png
```

---

## GPU Safety

- All models run on GPU when `--device cuda`.  
- If ASR fails with OOM:  
  - Segment is split  
  - Retried  
  - Results merged with confidence weighting  

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

You are free to:

- **Share** — copy and redistribute the material in any medium or format  
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially  

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/

