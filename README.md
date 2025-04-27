
# GibberJab NN &nbsp;🔊  
_A neural-network–powered latent-embedding “audio language” for machine-to-machine dialogue_

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GibberJab NN compresses natural-language text to ultra-compact byte strings, then transmits them as **frequency-shift-keyed (FSK) audio tones** via the open-source GGWave library.  
The current release implements a dictionary-based *TextCompressor*, adaptive chunking, and end-to-end acoustic transport with >95 % message recovery in typical indoor environments.

---

## Table&nbsp;of&nbsp;Contents
1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [How It Works](#how-it-works)  
6. [Repository Layout](#repository-layout)  
7. [Performance & Benchmarks](#performance--benchmarks)  
8. [Roadmap](#roadmap)  
9. [Citing or Contributing](#citing-or-contributing)

---

## Features
| Stage | Purpose | Highlights |
|-------|---------|------------|
| **Semantic Compression** | Replace high-frequency words / n-grams with single-byte codes | 1.2×–1.5× lossless ratio on conversational text |
| **Adaptive Chunker** | Fits compressed text into 125-byte GGWave payloads | Binary-search sizing + 10 % safety buffer |
| **FSK Modulator** | Converts each chunk to 48 kHz mono waveform | 6 tones ⟺ 3 bytes / frame, audible or ultrasonic |
| **Threaded Receiver** | Real-time capture, demodulation, and reassembly | Robust header `[i/N]`; partial recovery if chunks lost |

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-org>/GibberJab_NN.git
cd GibberJab_NN

# 2. Create & activate an isolated environment
conda create -n GibberJab_NN python=3.10 pip
conda activate GibberJab_NN

# 3. Install core requirements
pip install -r requirements.txt

# 4. Launch the main notebook (recommended workflow)
jupyter lab project.ipynb      # or: jupyter notebook project.ipynb
```

> **Tip:**  
> • `project.ipynb` walks through **send / receive / compression** end-to-end.  
> • If you prefer pure-CLI, you can still call `python send.py "Hello"` or `python receive.py --timeout 60`.

---

## Installation
<details>
<summary>Full dependency list</summary>

Most users only need `requirements.txt`; advanced experiments (VQ-VAE, BPE, etc.) pull in extra libraries.

```text
numpy>=1.21
torch>=1.10
transformers>=4.15
ggwave>=0.3.0
pyaudio
langgraph
matplotlib       # optional: plotting scripts
scikit-learn     # optional: analysis notebooks
```
</details>

---

## Usage

| Task | Command |
|------|---------|
| **Train / load** the compressor | `python neural_codec.py` |
| **Force retrain** from scratch | `python neural_codec.py --retrain` |
| **Batch-compress a text file** | `python tools/compress_file.py input.txt` |
| **Send a message over speakers** | `python demo_send.py "Hello, world!"` |
| **Listen & decode** for 60 s | `python demo_receive.py --timeout 60` |

Scripts are tiny wrappers around the same API you can import:

```python
from gibberjab import TextCompressor, transmission

compressor = TextCompressor.load("models/deep_compressor.pkl")
payload    = compressor.encode("Book a table at 7 pm, please.")
transmission.play_waveform(payload)
```

---

## How It Works

1. **TextCompressor**  
   *Scans a 10 M-character Project Gutenberg corpus, selects the top-512 patterns by space-savings, and stores a static codebook (ASCII 1–31 & 128–255).*  
2. **Adaptive Chunker**  
   *Binary-searches for the longest slice that compresses ≤ 125 bytes, prepends `[i/N]`, and queues for modulation.*  
3. **GGWave FSK Modulator**  
   *Maps each 4-bit nibble to an equally spaced tone (`F0 + k·46.875 Hz`), emits 6 tones per frame → 3 bytes / 20 ms.*  
4. **Threaded Receiver**  
   *Captures audio, performs FFT-based tone detection, Reed–Solomon error-check, reorders chunks, and decodes with the shared codebook.*  

> **Throughput**: 20–30 characters / s  **Recovery**: > 95 % under office-noise conditions

A concise block diagram lives in `docs/figs/GibberJab_Flow.png`.

---

## Repository Layout
```
GibberJab_NN/
├─ __pycache__/                # byte-code caches
├─ temp_audio/                 # scratch recordings
├─ temp_audio_client/          # client-side temp data
│
├─ .gitignore
├─ README.md                   # ← you’re reading it
├─ requirements.txt            # pip/conda deps
├─ secrets.toml                # 🔑 API keys ( NOT committed )
│
├─ project.ipynb               # ⭐ primary notebook walkthrough
├─ project_copy.ipynb          # spare / sandbox copy
│
├─ send.py                     # CLI sender (simple FSK demo)
├─ receive.py                  # CLI listener
├─ transmission.py             # high-level send/receive helpers
│
├─ NeuralTextCodec.py          # older stand-alone compressor script
├─ encoder_model.py            # experimental VQ-VAE encoder
├─ agent_flow.py               # LangGraph restaurant agent prototype
├─ client_bot.py               # conversational bot stub
│
├─ BERT_BPE_NTE.py             # BPE experiments
├─ BPE_NTE.py
├─ BPE_NTE_new.py
│
├─ corpus.txt                  # small sample corpus
├─ deep_corpus.txt             # full 10 M-char Gutenberg corpus
├─ deep_compressor.pkl         # pre-trained TextCompressor
│
├─ t2.py  test.py  test3.py  test4.py   # assorted scratch tests
└─ ecstatic-galaxy-457400-g3-93d1…png   # (screenshot asset)

---

## Performance & Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Compression Ratio | **1.2–1.5×** | Gutenberg + conversational text |
| Tx Rate | **20–30 char/s** | 48 kHz, default GGWave protocol |
| Message Recovery | **>95 %** | Indoor, laptop mic + speaker |
| Max Payload / Chunk | **125 bytes** | GGWave default; chunker auto-splits |

See `benchmarks/` for the reproducible Jupyter notebook.

---

## Roadmap
- 🔜 **Context-aware dictionaries** for domain-specific gains  
- 🔜 **BPE + VQ-VAE** encoder (projected 3× compression)  
- 🔜 **Forward-Error-Correction** for high-noise environments  
- 🔜 **Adaptive protocol selection** based on channel SNR  

Community PRs on any of the above are welcome!

---

## Citing or Contributing
If you use GibberJab NN in academic work, please cite:

```bibtex
@software{gibberjab2025,
  author       = {Hand, B. and Contributors},
  title        = {GibberJab_NN: Latent Audio Language Toolkit},
  year         = {2025},
  url          = {https://github.com/braydenhand/GibberJab_NN},
}
```

Contributions follow the standard GitHub flow:
1. Fork  → 2. Create feature branch  → 3. PR with tests  → 4. Code-review

Licensed under the MIT License – see [`LICENSE`](LICENSE) for details.
