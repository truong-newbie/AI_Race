# Quick Start - Train NER Model

## Option 1: Google Colab (Recommended)

1. Upload to Google Drive:
   - `notebooks/04_train_ner.ipynb`
   - Folder `src/`
   - Folder `data/processed/`

2. Open notebook, set Runtime → Change runtime type → **GPU**

3. Run first cell to install dependencies:
```python
!pip install torch transformers datasets tqdm accelerate
```

4. Run all cells

---

## Option 2: Local (with GPU)

```bash
# Install dependencies
pip install -r requirements_ner.txt

# Train
python scripts/train_ner.py --config configs/ner_xlmr_base.yaml

# Or with custom settings:
python scripts/train_ner.py \
    --config configs/ner_xlmr_base.yaml \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5
```

---

## Option 3: Evaluate Trained Model

```bash
# After training, evaluate on test set:
python scripts/evaluate_ner.py \
    --model_path outputs/ner_xlmr_base/best_model.pt \
    --data_path data/processed/internal_test.jsonl

# Or predict on new text:
python scripts/predict_ner.py \
    --model_path outputs/ner_xlmr_base/best_model.pt \
    --text "Bệnh nhân ho sốt, xét nghiệm máu bình thường"
```

---

## Output Location

Trained models saved to: `outputs/ner_xlmr_base/`
- `best_model.pt` - Best checkpoint
- `evaluation_metrics.json` - Metrics
- `training_history.png` - Loss/F1 curves
