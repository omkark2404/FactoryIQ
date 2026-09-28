# FactoryIQ System Architecture

## 1. Executive Summary

FactoryIQ is a unified manufacturing quality intelligence platform designed for quality engineers and plant operations personnel. It fuses visual inspection streams, machine sensor log analytics, and technical documentation via Retrieval-Augmented Generation (RAG).

---

## 2. End-to-End Pipeline Workflow

```text
FACTORYIQ
│
├── 1. Product Inspection Images (Bottle, Screw, Metal Nut, Tile)
│   └─► PyTorch Vision Model (ResNet Backbone + Anomaly Scorer) ──► Anomaly Score & Region Mask
│
├── 2. Machine Sensor Telemetry (Temp, Pressure, Speed, Shift)
│   └─► Scikit-Learn Quality Risk Model (RandomForest) ──────────► HIGH/LOW Risk & Defect Rate %
│
└── 3. Technical Documents & Inspection Reports (PDF / Images)
    └─► OCR Engine ──► Clean Text ──► Chunking ──► Embeddings ──► Vector DB ──► RAG Retriever
                                                                                     │
                                                                                     ▼
                                                                        LLM Quality Assistant
                                                                        (With Source Citations)
```

---

## 3. RAG Pipeline Mechanics

The FactoryIQ RAG engine follows a strict multi-step document lifecycle:

1. **OCR Text Extraction**: Raw inspection PDFs or scanned report images pass through `pdfplumber`/`pytesseract` to extract plaintext.
2. **Text Cleaning & Parsing**: Normalizes formatting, strips header clutter, and parses key metadata tags (`batch_id`, `machine_id`, `operating_temp_c`, `shift`).
3. **Overlapping Sentence Chunking**: Applies sliding window chunker (350 words, 60 word overlap), ensuring metadata is attached to every individual chunk.
4. **Dense Embedding Vectorization**: Generates 384-dimensional vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
5. **Vector Store & Cosine Similarity Search**: Indexes chunk vectors into `VectorStore` with support for metadata filtering.
6. **Retriever & Evidence Synthesis**: Fetches top-k document chunks and builds system prompt combining vision results, ML risk predictions, SOP limits, and historical incident logs.
7. **Source Citation**: Every response lists verified citations (e.g. `✓ Inspection Report RB-2041`, `✓ Press-04 SOP`, `✓ Previous Incident #018`).

---

## 4. Computer Vision Anomaly Detection

- Target categories locked to 4 industrial categories: **Bottle**, **Screw**, **Metal Nut**, **Tile**.
- Model uses feature representation distance / reconstruction error to assign normalized anomaly scores [0.0 - 1.0].
- High anomaly scores (> 0.65) trigger visual defect region tag annotations.

---

## 5. Classical ML Quality Risk Predictor

- Trained on synthetic production data (`data/synthetic/production_data.csv`).
- Features: `temperature_c`, `pressure_psi`, `line_speed_mmin`, `shift_encoded`, `previous_defects`.
- Models: `RandomForestClassifier` for risk level (`HIGH` / `LOW`) and `RandomForestRegressor` for expected defect rate.
