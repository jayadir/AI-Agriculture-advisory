# LIGN Model Setup

## ⚠️ IMPORTANT: Model File Required

The LIGN model file is **NOT included** in the repository. You must:

### Option 1: Train the Model (Recommended)
If you have the training notebook or script:
```bash
# Run training (example - adjust based on your setup)
cd notebooks
jupyter notebook lign-model.ipynb  # Or run training script
```

Expected output: `lign_best_model.pth` (~250MB)

### Option 2: Copy from Backup
If you have a pre-trained model:
```bash
# Copy from backup location
cp /path/to/backup/lign_best_model.pth artifacts/models/
```

### Option 3: Use Without LIGN (Fallback)
Disable LIGN temporarily:
```bash
export LIGN_ENABLED=0
python app/main.py
```

## Model Location
Place the trained model at:
```
agent_service/
  artifacts/
    models/
      lign_best_model.pth  ← Place here
```

## Verification
Test if the model loads correctly:
```bash
cd agent_service
python tests/test_lign_reranking.py
```

Expected output:
```
[LIGN] Model loaded from artifacts/models/lign_best_model.pth
LIGN Re-ranker Enabled (k=20, top_k=5, threshold=0.3)
```

## Training Requirements
If training from scratch:
- Dataset: MS MARCO passage ranking
- Hardware: GPU recommended (8GB+ VRAM)
- Time: ~4-6 hours on single GPU
- Dependencies: Already in `requirements.txt`

## Next Steps
Once model is in place:
1. Run test: `python tests/test_lign_reranking.py`
2. Start service: `python app/main.py`
3. Verify logs show: `[LIGN] Re-ranking X candidates...`
