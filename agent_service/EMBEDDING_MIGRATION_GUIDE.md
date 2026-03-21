# Embedding Model Migration Guide: Jina → BGE

## Overview

This project now supports **two embedding models** that can be switched via environment variables:

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| **Jina-Embeddings-v3** | 1024-dim | High accuracy, larger model |
| **BAAI/bge-small-en-v1.5** | 384-dim | Fast, efficient, smaller model |

## Quick Start

### 1. Switch to BGE Embeddings

Set the environment variable in PowerShell:
```powershell
$env:EMBEDDING_MODEL="bge"
```

Or add to your `.env` file:
```bash
EMBEDDING_MODEL=bge
```

### 2. Rebuild FAISS Index

Run the rebuild script:
```powershell
cd agent_service
python rebuild_faiss_bge.py
```

The script will:
- ✅ Load existing documents from your current FAISS index
- ✅ Re-embed using BGE model (384-dim)
- ✅ Save new index to `artifacts/vector_db/agri_faiss_index_bge`
- ✅ Provide instructions for replacing the old index

### 3. Replace Index

After successful rebuild:
```powershell
# Backup old index (optional)
Move-Item "artifacts/vector_db/agri_faiss_index" "artifacts/vector_db/agri_faiss_index_backup"

# Use new BGE index
Move-Item "artifacts/vector_db/agri_faiss_index_bge" "artifacts/vector_db/agri_faiss_index"
```

### 4. Start Service

```powershell
python -m app.main
```

The service will automatically load:
- ✅ BGE embedder (384-dim)
- ✅ BGE query expander from `artifacts/models/query_expander.pth`
- ✅ BGE-compatible FAISS index

## Architecture

### Code Changes

#### 1. **config.py** - Environment Configuration
```python
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jina")
```

#### 2. **embeddings.py** - Dual Embedder Support
```python
class BGEEmbedder(Embeddings):
    """BAAI/bge-small-en-v1.5 (384-dim)"""
    # Loads BAAI/bge-small-en-v1.5
    
class JinaEmbedder(Embeddings):
    """Jina-Embeddings-v3 (1024-dim)"""
    # Loads jinaai/jina-embeddings-v3

def get_embedder():
    """Returns correct embedder based on EMBEDDING_MODEL env var"""
    if settings.EMBEDDING_MODEL.lower() == "bge":
        return BGEEmbedder()
    else:
        return JinaEmbedder()
```

#### 3. **query_expander_bge.py** - New 384-dim Expander
```python
class DeepResidualExpanderBGE(nn.Module):
    """Query Expander for BGE embeddings (384-dim)"""
    def __init__(self, input_dim=384, hidden_dim=512):
        # 4 expansion heads: para, broad, tech, expl
```

#### 4. **engine.py** - Dynamic Model Loading
```python
if settings.EMBEDDING_MODEL.lower() == "bge":
    self.expander = DeepResidualExpanderBGE(input_dim=384)
    expander_path = "artifacts/models/query_expander.pth"
else:
    self.expander = DeepResidualExpander(input_dim=1024)
    expander_path = "artifacts/models/deep_residual_expander.pt"
```

## Model Files Required

### For BGE (384-dim)
```
artifacts/
├── models/
│   └── query_expander.pth           # Your trained 384-dim expander
└── vector_db/
    └── agri_faiss_index/            # BGE-embedded documents
```

### For Jina (1024-dim)
```
artifacts/
├── models/
│   └── deep_residual_expander.pt    # Existing 1024-dim expander
└── vector_db/
    └── agri_faiss_index/            # Jina-embedded documents
```

## Environment Variables

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `EMBEDDING_MODEL` | `jina` / `bge` | `jina` | Embedding model selection |
| `FORCE_CPU` | `0` / `1` | `0` | Force CPU mode |
| `LIGN_ENABLED` | `0` / `1` | `1` | Enable LIGN re-ranker |

## Performance Comparison

| Metric | Jina (1024-dim) | BGE (384-dim) |
|--------|----------------|---------------|
| Model Size | ~560 MB | ~133 MB |
| Embedding Speed | ~50ms/query | ~20ms/query |
| Memory Usage | ~2GB GPU | ~500MB GPU |
| Accuracy | High | Good |
| Recommendation | High-accuracy tasks | Production deployment |

## Troubleshooting

### Issue: Documents not loading from existing index

**Solution**: The rebuild script tries to extract documents from your existing FAISS index. If this fails, you can load from JSONL:

```python
# Edit rebuild_faiss_bge.py
jsonl_paths = [
    "path/to/your/documents.jsonl",
]
```

### Issue: Model dimension mismatch

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solution**: Ensure:
1. `EMBEDDING_MODEL=bge` is set
2. Using `query_expander.pth` (384-dim weights)
3. FAISS index was rebuilt with BGE

### Issue: Query expander not found

**Error**: `Query Expander weights not found at artifacts/models/query_expander.pth`

**Solution**: 
1. Ensure you have the trained 384-dim model weights
2. Place at `artifacts/models/query_expander.pth`
3. Model should match the `DeepResidualExpanderBGE` architecture

## Switching Back to Jina

To revert to Jina embeddings:

```powershell
# Set environment
$env:EMBEDDING_MODEL="jina"

# Restore original index (if backed up)
Move-Item "artifacts/vector_db/agri_faiss_index_backup" "artifacts/vector_db/agri_faiss_index"

# Restart service
python -m app.main
```

## Migration Checklist

- [ ] Train 384-dim query expander model (if not done)
- [ ] Save model as `artifacts/models/query_expander.pth`
- [ ] Set `EMBEDDING_MODEL=bge` in environment
- [ ] Run `rebuild_faiss_bge.py`
- [ ] Backup old FAISS index
- [ ] Replace with new BGE index
- [ ] Test queries with new embeddings
- [ ] Monitor performance and accuracy

## Support

If you encounter issues:
1. Check environment variables are set correctly
2. Verify model files exist in `artifacts/models/`
3. Ensure FAISS index dimension matches embedding model
4. Check logs for dimension mismatch errors
