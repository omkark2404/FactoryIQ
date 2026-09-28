# FactoryIQ — AI-Powered Manufacturing Quality Intelligence System

> **Industrial Quality Intelligence Engine combining Computer Vision, Document OCR, Retrieval-Augmented Generation (RAG), Classical ML Risk Prediction, and FastAPI.**

---

## 📌 Overview

**FactoryIQ** is an integrated manufacturing quality intelligence platform designed for quality engineering teams. In industrial production lines, quality teams process multimodal data sources—product inspection images, machine sensor logs, equipment SOPs, inspection PDFs, and past incident reports. 

FactoryIQ fuses these data streams into a unified intelligence system to answer:
> **"What is wrong with this batch, why did it happen, and what should the quality engineer check next?"**

---

## 🏗️ System Architecture

```text
                                  FACTORYIQ
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
  Product Images               Inspection PDFs              Production Data
 (Bottle, Screw,                 & SOP Files                 (Temp, Speed, 
 Metal Nut, Tile)                     │                      Pressure, etc.)
         │                            ▼                            │
         ▼                           OCR                           ▼
  PyTorch Vision Model                │                        Scikit-Learn
  (Anomaly Scorer &                   ▼                      Risk Predictor
   Category Detection)       Document Processing                   │
         │                   (Clean & Metadata)                    │
         │                            │                            │
         │                            ▼                            │
         │                         Chunking                        │
         │                            │                            │
         │                            ▼                            │
         │                        Embeddings                       │
         │                            │                            │
         │                            ▼                            │
         │                      Vector Database                    │
         │                            │                            │
         │                            ▼                            │
         │                           RAG ──────────────────────────┤
         │                            │                            │
         ▼                            ▼                            ▼
   Defect Result              Retrieved Context              ML Risk Prediction
 (ANOMALY / NORMAL)          (SOPs, Incident Logs)           (HIGH / LOW Risk)
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                                 FastAPI Backend
                                      │
                                      ▼
                             LLM Quality Assistant
                                      │
                                      ▼
                             FactoryIQ Dashboard
```

---

## 🛠️ Key Components & Methodology

### 1. Computer Vision (PyTorch)
- **Target Categories**: 4 core industrial inspection targets: `bottle`, `screw`, `metal_nut`, `tile`.
- **Dataset**: Built on [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) (Anomaly Detection dataset).
- **Functionality**: Extracts visual feature embeddings using PyTorch, calculates anomaly scores against normal baselines, and generates defect highlight masks.

### 2. OCR + Document Processing
- **Pipeline**: PDF/Image Document → OCR Extraction → Text Cleaning → Key Field Parsing (Batch ID, Machine ID, Temperature, Shift, Defect Count).
- **Metadata Tagging**: Preserves contextual tags (`document_type`, `batch_id`, `machine_id`, `shift`, `date`).

### 3. RAG Engine
- **Chunking**: Overlapping sliding window chunker preserving document hierarchy and metadata.
- **Embeddings & Vector Store**: Dense embedding generation (`sentence-transformers/all-MiniLM-L6-v2`) with cosine similarity vector search & metadata filtering.
- **Retriever & Citation Engine**: Fetches relevant SOPs, historical incident logs, and manual guidelines, providing verifiable source attribution.

### 4. Classical ML Quality Risk Model (Scikit-Learn)
- **Features**: Batch operational data (`temperature`, `pressure`, `line_speed`, `shift`, `previous_defects`).
- **Prediction**: Random Forest / Gradient Boosting classifier predicting **High/Low Rejection Risk** & estimated defect rate.

### 5. FastAPI & Interactive Dashboard
- REST API layer connecting vision inference, ML risk prediction, document ingestion, and RAG QA.
- Web dashboard displaying batch overview, inspection results, uploaded PDF highlights, and grounded LLM recommendations with citation badges.

---

## 📂 Project Structure

```text
FactoryIQ/
├── data/
│   ├── raw/                  # MVTec dataset placeholders & raw documents
│   ├── processed/            # Extracted OCR text, chunks & vector indices
│   └── synthetic/            # Synthetic production CSVs & incident logs
├── notebooks/                # Data exploration, CV training, ML risk & RAG eval
├── src/
│   ├── vision/               # PyTorch dataset, model, training & inference
│   ├── ocr/                  # Document OCR extraction, cleaning & field parser
│   ├── rag/                  # Loader, chunker, embeddings, vector store & pipeline
│   ├── ml/                   # Preprocessing, ML risk model training & prediction
│   └── api/                  # FastAPI app, schemas, and routes
├── models/                   # Saved vision & ML risk model weights
├── frontend/                 # Interactive HTML/JS Dashboard
├── vector_store/             # Persisted vector database files
├── tests/                    # Comprehensive unit tests for OCR, RAG, ML, API
└── docs/                     # Architecture & Dataset documentation
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/your-username/FactoryIQ.git
cd FactoryIQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
cp .env.example .env
```

### 3. Train Quality Risk Model & Ingest Documents
```bash
# Train ML quality risk model on synthetic production data
python -m src.ml.train

# Ingest sample manufacturing SOPs & incident documents into Vector DB
python -m src.rag.pipeline --ingest
```

### 4. Launch Application
```bash
# Start FastAPI backend server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/dashboard` in your web browser.

---

## 📊 Evaluation & Testing

Run unit tests across OCR, RAG, ML risk predictor, and API routes:
```bash
pytest tests/ -v
```

---

## ⚠️ Disclaimer
- Data files in `data/synthetic/` are synthetically generated for demonstration & industrial testing purposes.
- Computer vision module pre-configured for MVTec AD categories (`bottle`, `screw`, `metal_nut`, `tile`).
