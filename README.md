# BPE-Trans: Low-Resource Neural Machine Translation Mini Project
**Course**: CLNLP Mini Project 2  
**Topic**: Low-Resource Neural Machine Translation with Byte-Pair Encoding (BPE) & Seq2Seq Transformer

---

## 📖 1. Project Overview
In Natural Language Processing, low-resource languages face two main bottlenecks:
1. **Severe Data Scarcity**: Training massive large language models (LLMs) requires millions/billions of tokens.
2. **High Out-of-Vocabulary (OOV) Rate**: Word-level tokenizers fail on rare or unseen morphological word variants.

**BPE-Trans** addresses this by combining:
- **Subword Byte-Pair Encoding (BPE)**: Pure Python BPE algorithm that segments compound words and suffixes into reusable subword units, directly handling rare and OOV words in indigenous dialects.
- **Lightweight PyTorch Seq2Seq Transformer**: Compact multi-head self-attention encoder-decoder with sinusoidal positional encodings, optimized to train and infer quickly on standard CPUs.
- **Modern Lightweight Web UI**: Interactive FastAPI web app styled with Tailwind CSS, displaying real-time translation, BPE subword segmentation breakdown, confidence scores, and checkpoint benchmarks.

---

## 🗂️ 2. Supported Low-Resource Language Datasets
All datasets from **Category I** and **Category II** are supported:
- **Category I**: Assamese, Khasi, Manipuri, Meitei-Mayek, Mizo, Nyishi.
- **Category II**: Karbi, Kokborok, Nagamese, Bodo, Tagin.

---

## 🚀 3. Quick Start Guide

### Step 1: Install Dependencies
Standard packages required (already present in standard Python 3.10+ environments):
```bash
pip install torch fastapi uvicorn jinja2 openpyxl pandas
```

### Step 2: Run the Web Application
Launch the web interface locally:
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🔬 4. Training Models for Other Dialects
You can train a model directly via CLI or via the web dashboard:
```bash
# Train on Nagamese
python -m src.train --lang nagamese --epochs 10

# Or train Karbi
python -m src.train --lang karbi --epochs 10
```
Trained checkpoints and BPE vocabularies are saved automatically under `models_saved/<lang_key>/`.

---

## 📁 5. Directory Structure
```
CLNLP Mini Project 2/
├── Category I/               # Provided datasets (Assamese, Mizo, Khasi, etc.)
├── Category II/              # Provided datasets (Karbi, Nagamese, Kokborok, etc.)
├── models_saved/             # Saved model checkpoints & BPE vocabularies
├── src/
│   ├── app.py                # FastAPI backend & API routing
│   ├── data_loader.py        # Streamlined dataset parser & cleaner
│   ├── inference.py          # Translation engine & confidence scoring
│   ├── model.py              # PyTorch Seq2Seq Transformer implementation
│   ├── tokenizer.py          # Pure Python Subword BPE Tokenizer
│   └── train.py              # Training & BLEU evaluation loop
├── templates/
│   └── index.html            # Modern Tailwind CSS interactive frontend
├── run.py                    # One-click execution script
└── README.md                 # Project documentation
```
