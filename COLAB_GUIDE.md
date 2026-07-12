# Hướng dẫn sử dụng Google Colab

## Cấu trúc thư mục trên Google Drive

```
MyDrive/
└── medical_ontology/
    ├── src/                   ← mã nguồn (git clone vào đây)
    ├── data/
    │   ├── processed/         ← train.jsonl, dev.jsonl
    │   ├── raw/               ← test data
    │   └── synthetic/         ← (optional) thêm dữ liệu huấn luyện
    ├── checkpoints/
    │   └── ner/               ← best_model.pt, latest_checkpoint.pt
    │                          ← ⚠️ Lớn — lưu trong Drive
    └── outputs/
        └── ner/               ← eval_metrics.json, training_history.png
```

## Bước 0: Upload data

1. Mount Google Drive trên Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Upload files:
   - `data/processed/train.jsonl` → `MyDrive/medical_ontology/data/processed/`
   - `data/processed/dev.jsonl` → `MyDrive/medical_ontology/data/processed/`
   - Test files → `MyDrive/medical_ontology/data/raw/`

3. Clone repo:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/medical-ontology.git /content/drive/MyDrive/medical_ontology
   ```

## Bước 1: Huấn luyện NER (Notebook 04)

**File:** `04_train_ner.ipynb`

```bash
# Chạy tuần tự các cells:
# [0] Install dependencies
# [1] Configuration          ← ĐIỀU CHỈNH batch_size theo VRAM
# [2] GPU check
# [3] Mount Drive
# [4] Import from src/
# [5] Dataset preview
# [6] Load tokenizer & datasets
# [7] Init model + optimizer + scheduler
# [8] Checkpoint resume       ← Tự động resume nếu có checkpoint
# [9] ★ Training loop         ← Chạy cell này để train
# [10] Final evaluation
# [11] Training history plot
# [12] Inference demo
# [13] Backup checkpoint lên Drive
# [14] Free GPU memory
```

**Cấu hình cho T4 (Colab free, ~6GB VRAM):**
```python
NOTEBOOK_CFG = {
    "train_batch_size": 8,           # T4 OK
    "gradient_accumulation_steps": 1,
    "fp16": True,
}
```

**Cấu hình cho GPU yếu (2-4GB VRAM):**
```python
NOTEBOOK_CFG = {
    "train_batch_size": 4,           # Yếu
    "gradient_accumulation_steps": 4,  # effective_batch = 16
    "fp16": True,
}
```

**Checkpoint tự động:**
- `latest_checkpoint.pt` — lưu mỗi epoch (resume được)
- `best_model.pt` — lưu khi có best F1 mới
- Backup tự động lên Drive ở cell [13]

## Bước 2: Build Retrieval Embeddings (Notebook 05)

**File:** `05_build_retrieval.ipynb`

```bash
# [0-3] Setup (như bước 1)
# [4] Build ICD-10 dense embeddings
#        → Cache: .cache/icd_dense/embeddings_*.npy
#        → Lần sau: đọc từ cache, không re-encode
# [5] Test ICD retrieval
# [6] Build RxNorm dense embeddings
#        → Cache: .cache/rxnorm_dense/embeddings_*.npy
# [7] Test RxNorm retrieval
# [8-9] Verify cache
# [10] Free GPU memory
```

**Lần chạy đầu:** ~2-3 phút (encode ~38 ICD + ~28 RxNorm entries)
**Lần chạy sau:** <5 giây (đọc từ cache)

## Bước 3: Phân tích lỗi (Notebook 06)

**File:** `06_error_analysis.ipynb`

```bash
# Chạy sau khi có outputs/errors.csv từ evaluation
# [0-1] Setup + load data
# [2] Error type distribution (pie chart)
# [3] Errors by entity type (bar chart)
# [4] Most frequently missed entities
# [5] Most frequently false positives
# [6] Wrong type analysis
# [7] Per-class F1 bar chart
# [8] E2E metrics by level
# [9] Generate error_analysis_report.md
# [10] Export errors_for_review.csv
```

**Output files:**
- `outputs/error_type_distribution.png`
- `outputs/errors_by_entity_type.png`
- `outputs/top_missed_entities.png`
- `outputs/top_false_positives.png`
- `outputs/per_class_metrics.png`
- `outputs/e2e_metrics_by_level.png`
- `outputs/error_analysis_report.md`
- `outputs/errors_for_review.csv`

## Bước 4: Inference & Generate Submission (Notebook 07)

**File:** `07_generate_submission.ipynb`

```bash
# [0-2] Setup + GPU check + mount Drive
# [3] Import from src/
# [4] Load NER checkpoint    ← Tự động load best_model.pt từ Drive
# [5] Build pipeline         ← rule-based + NER model
# [6-7] Find & read test data
# [8] ★ Batch inference       ← Chunk long text, torch.no_grad()
# [9] Convert to submission format
# [10] Save predictions.jsonl
# [11] Generate submission.zip
# [12] Summary stats
# [13] Free GPU memory
```

**Output:**
- `outputs/predictions.jsonl` — tất cả predictions
- `outputs/submission.zip` — nén submission

## Memory Optimization Cheatsheet

| Vấn đề | Giải pháp |
|--------|-----------|
| OOM khi train | Giảm `train_batch_size`, tăng `gradient_accumulation_steps` |
| OOM khi inference | Dùng `torch.no_grad()`, chunk long text |
| Chậm trên CPU | Chỉ dùng GPU, bật `fp16=True` |
| Embeddings re-encode mỗi lần | Cache ở `.cache/icd_dense/` và `.cache/rxnorm_dense/` |
| VRAM không giải phóng | Gọi `gc.collect()` + `torch.cuda.empty_cache()` |
| Nhiều model cùng lúc | Load từng model, giải phóng trước khi load model mới |

## Thứ tự chạy khuyến nghị

```
1. 04_train_ner.ipynb    → Huấn luyện NER (1-2 giờ với T4)
2. 05_build_retrieval.ipynb  → Tạo embeddings (2-3 phút)
3. [Chạy evaluation ở local/script]
4. 06_error_analysis.ipynb    → Phân tích lỗi
5. 07_generate_submission.ipynb → Tạo submission
```

## Troubleshooting

**Lỗi "CUDA out of memory":**
```python
# Trong config cell:
NOTEBOOK_CFG["train_batch_size"] = 4
NOTEBOOK_CFG["gradient_accumulation_steps"] = 4
# Sau đó restart runtime và chạy lại từ đầu
```

**Lỗi "Module not found":**
```python
# Kiểm tra sys.path:
import sys
print(sys.path)
# Đảm bảo PROJECT_ROOT đã được insert:
sys.path.insert(0, '/content/drive/MyDrive/medical_ontology')
```

**Không có checkpoint để resume:**
- Kiểm tra `checkpoints/ner/best_model.pt` có tồn tại không
- Đường dẫn phải đúng: `/content/drive/MyDrive/medical_ontology/checkpoints/ner/`

**VRAM check:**
```python
import torch
print(torch.cuda.get_device_properties(0).total_mem / 1e9, "GB")
```
