# FactoryIQ Dataset Guide & Setup Strategy

FactoryIQ combines three distinct datasets to model industrial manufacturing inspection and quality retrieval.

---

## 1. Vision Dataset — MVTec AD (Reduced Scope)

For the visual inspection component, FactoryIQ targets 4 representative industrial product categories:

| Category | Inspection Type | Common Defect Types |
| :--- | :--- | :--- |
| **Bottle** | Surface & shape defects | Linear cracks, contamination, broken neck |
| **Screw** | Structural & thread defects | Thread abnormality, surface scratch, bent head |
| **Metal Nut** | Component surface defects | Metal burr, scratch, perimeter dent |
| **Tile** | Surface texture defects | Crack, surface stain, gray stroke |

### Download Setup
Download MVTec AD category archives from [MVTec AD Official Page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and extract into `data/raw/mvtec/`:

```text
data/raw/mvtec/
├── bottle/
├── screw/
├── metal_nut/
└── tile/
```

*Note: FactoryIQ includes a synthetic image generator fallback in `src/vision/dataset.py` so the vision pipeline can run immediately without full dataset downloads.*

---

## 2. Document & OCR Pipeline Dataset

- Primary validation benchmark: [DocVQA Dataset](https://site.docvqa.org/datasets/docvqa) for document QA validation.
- FactoryIQ Synthetic Manufacturing Set:
  - `quality_manual.txt`: Escalation thresholds, SOP guidelines, and thermal limits.
  - `machine_sop.txt`: Specific operating procedures for Press-04.
  - `inspection_report_RB2041.txt`: Actual inspection report for Batch RB-2041.
  - `incident_report_018.txt`: Historical defect incident INC-018 detailing thermal overheating issues on Press-04.

---

## 3. Production Quality Risk Dataset (`data/synthetic/production_data.csv`)

Synthetic production log containing 15 historical batch runs with metrics:
- `batch_id`: Unique batch identifier (e.g. `RB-2041`).
- `machine_id`: Press unit (`Press-01` to `Press-04`).
- `temperature_c`: Operating temperature in °C.
- `pressure_psi`: Operating hydraulic pressure.
- `line_speed_mmin`: Production line speed.
- `shift`: Operating shift (A, B, C).
- `previous_defects`: Count of historical machine defects.
- `defect_rate_pct`: Recorded percentage defect rate.
- `quality_risk_level`: Classified risk level (`HIGH` / `LOW`).
