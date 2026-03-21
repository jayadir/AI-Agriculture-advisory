# BGE Embedding Quick Reference

## Setup Commands

```powershell
# 1. Set embedding model to BGE
$env:EMBEDDING_MODEL="bge"

# 2. Verify setup
python verify_setup.py

# 3. Rebuild FAISS with BGE embeddings
python rebuild_faiss_bge.py

# 4. Replace old index
Move-Item "artifacts/vector_db/agri_faiss_index" "artifacts/vector_db/agri_faiss_index_backup"
Move-Item "artifacts/vector_db/agri_faiss_index_bge" "artifacts/vector_db/agri_faiss_index"

# 5. Start service
python -m app.main
```

## File Structure

```
agent_service/
├── .env.example                          # Environment config template
├── EMBEDDING_MIGRATION_GUIDE.md          # Full migration guide
├── rebuild_faiss_bge.py                  # FAISS rebuild script
├── verify_setup.py                       # Setup verification script
├── app/
│   ├── core/
│   │   └── config.py                     # ✅ Updated: EMBEDDING_MODEL env var
│   ├── rag/
│   │   ├── embeddings.py                 # ✅ Updated: BGEEmbedder + dynamic get_embedder()
│   │   ├── query_expander.py             # Jina expander (1024-dim)
│   │   ├── query_expander_bge.py         # ✅ New: BGE expander (384-dim)
│   │   └── engine.py                     # ✅ Updated: Dynamic model loading
│   └── ...
└── artifacts/
    ├── models/
    │   ├── query_expander.pth            # Your 384-dim BGE model (required)
    │   └── deep_residual_expander.pt     # Existing 1024-dim Jina model
    └── vector_db/
        └── agri_faiss_index/             # FAISS index (rebuild for BGE)
```

## Environment Variables

```bash
# Primary configuration
EMBEDDING_MODEL=bge              # "jina" or "bge"
FORCE_CPU=0                      # 0 (GPU) or 1 (CPU)
LIGN_ENABLED=1                   # 0 (off) or 1 (on)

# Database (unchanged)
MONGO_URL=mongodb://localhost:27017
DB_NAME=agri_brain_db
```

## Model Dimensions

| Model | Embedding Dim | Expander Class | Weight File |
|-------|--------------|----------------|-------------|
| Jina v3 | 1024 | `DeepResidualExpander` | `deep_residual_expander.pt` |
| BGE Small | 384 | `DeepResidualExpanderBGE` | `query_expander.pth` |

## Verification Checklist

- [ ] `$env:EMBEDDING_MODEL="bge"` set
- [ ] `artifacts/models/query_expander.pth` exists
- [ ] `python verify_setup.py` passes all checks
- [ ] FAISS index rebuilt with BGE
- [ ] Service starts without dimension errors
- [ ] Test query returns results

## Common Issues

### Dimension Mismatch
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1024 and 384x512)
```
**Fix**: FAISS index doesn't match embedding model. Rebuild with `rebuild_faiss_bge.py`

### Model Not Found
```
Query Expander weights not found at artifacts/models/query_expander.pth
```
**Fix**: Ensure you have the trained 384-dim model weights

### Wrong Embedder Loaded
```
📊 Using Jina-Embeddings-v3 (1024-dim) embeddings
```
**Fix**: Set `$env:EMBEDDING_MODEL="bge"` before starting service

## Switching Between Models

### Switch to BGE
```powershell
$env:EMBEDDING_MODEL="bge"
python verify_setup.py
python -m app.main
```

### Switch to Jina
```powershell
$env:EMBEDDING_MODEL="jina"
python verify_setup.py
python -m app.main
```

## Performance Tips

1. **Use BGE for production**: 3x faster, 75% smaller memory footprint
2. **Use Jina for accuracy**: Higher dimensional space, better for complex queries
3. **Enable LIGN re-ranking**: Improves relevance for both models
4. **Force CPU if low VRAM**: Set `FORCE_CPU=1` for GPUs with <4GB VRAM

## Testing

```python
# Quick test
from app.rag.embeddings import get_embedder

embedder = get_embedder()
emb = embedder.embed_query("test query")
print(f"Dimension: {len(emb)}")  # Should be 384 for BGE
```

## Support Files

- `EMBEDDING_MIGRATION_GUIDE.md` - Full migration documentation
- `verify_setup.py` - Automated setup verification
- `rebuild_faiss_bge.py` - FAISS index rebuild script
- `.env.example` - Environment configuration template
