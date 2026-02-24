"""
Sliding window text chunker with overlap for LIGN scoring.
This prevents information loss at chunk boundaries.
"""
from typing import List, Tuple
import re


def sliding_window_chunks(
    text: str,
    window_size: int = 512,
    overlap: int = 128,
    tokenizer=None
) -> List[Tuple[str, int, int]]:
    """
    Create overlapping text chunks using sliding window.
    
    Args:
        text: Input text to chunk
        window_size: Window size in tokens (or chars if no tokenizer)
        overlap: Overlap size in tokens (or chars)
        tokenizer: Optional tokenizer (HuggingFace), if None uses char-based
        
    Returns:
        List of (chunk_text, start_pos, end_pos) tuples
    """
    if not text or len(text) < 50:
        return []
    
    # Use tokenizer if provided, otherwise char-based
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= window_size:
            return [(text, 0, len(text))]
        
        chunks = []
        stride = window_size - overlap
        
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i + window_size]
            
            if len(window_tokens) < overlap // 2:  # Skip tiny trailing chunks
                break
            
            chunk_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
            chunks.append((chunk_text, i, i + len(window_tokens)))
            
            # Stop if we've covered the text
            if i + window_size >= len(tokens):
                break
        
        return chunks
    
    else:
        # Char-based sliding window (simpler, no tokenizer needed)
        # Use sentence-aware windowing to avoid mid-sentence cuts
        sentences = split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        sentence_idx = 0
        
        while sentence_idx < len(sentences):
            current_text = ""
            temp_idx = sentence_idx
            
            # Build window by adding sentences until we reach window_size chars
            while temp_idx < len(sentences) and len(current_text) < window_size:
                current_text += sentences[temp_idx] + " "
                temp_idx += 1
            
            current_text = current_text.strip()
            
            if len(current_text) >= 50:  # Min chunk size
                # Calculate actual start position in original text
                start_pos = sum(len(s) + 1 for s in sentences[:sentence_idx])
                end_pos = start_pos + len(current_text)
                chunks.append((current_text, start_pos, end_pos))
            
            # Calculate how many sentences to keep as overlap
            overlap_sentences = 0
            overlap_chars = 0
            for s in reversed(sentences[sentence_idx:temp_idx]):
                if overlap_chars + len(s) <= overlap:
                    overlap_chars += len(s) + 1
                    overlap_sentences += 1
                else:
                    break
            
            # Move forward: if we only processed 1 sentence, move by 1, else keep overlap
            if temp_idx - sentence_idx <= 1:
                sentence_idx += 1
            else:
                sentence_idx = temp_idx - max(1, overlap_sentences)
            
            # Safety: prevent infinite loop
            if sentence_idx >= len(sentences) or temp_idx >= len(sentences):
                break
        
        return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter that handles common abbreviations.
    """
    # Handle common abbreviations
    text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
    text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
    text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
    text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
    text = re.sub(r'\betc\.', 'etc<DOT>', text)
    text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
    text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


def deduplicate_overlapping_chunks(
    scored_chunks: List[Tuple[str, float, int, int]],
    score_threshold: float = 0.4
) -> List[Tuple[str, float]]:
    """
    Remove overlapping chunks, keeping highest-scored version of overlapping regions.
    
    Args:
        scored_chunks: List of (text, score, start_pos, end_pos)
        score_threshold: Minimum score to keep
        
    Returns:
        List of (text, score) deduplicated chunks
    """
    # Filter by threshold first
    filtered = [(text, score, start, end) for text, score, start, end in scored_chunks 
                if score >= score_threshold]
    
    if not filtered:
        return []
    
    # Sort by score descending
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    kept_chunks = []
    kept_ranges = []
    
    for text, score, start, end in filtered:
        # Check if this chunk overlaps significantly with any kept chunk
        overlaps = False
        for kept_start, kept_end in kept_ranges:
            overlap_start = max(start, kept_start)
            overlap_end = min(end, kept_end)
            overlap_len = max(0, overlap_end - overlap_start)
            
            chunk_len = end - start
            # If more than 70% overlap, skip this chunk
            if overlap_len / chunk_len > 0.7:
                overlaps = True
                break
        
        if not overlaps:
            kept_chunks.append((text, score))
            kept_ranges.append((start, end))
    
    return kept_chunks


# Example usage and testing
if __name__ == "__main__":
    sample_text = """
    The red flour beetle (Tribolium castaneum) is a major pest of stored grain products.
    It infests flour, cereals, and other dried food products in homes and warehouses.
    Control methods include fumigation with phosphine gas and proper sanitation.
    Natural predators like parasitic wasps can also help control beetle populations.
    Heat treatment at 55°C for 24 hours is an effective non-chemical alternative.
    Prevention involves sealing storage containers and maintaining low humidity levels.
    Regular inspection of stored products can catch infestations early.
    """
    
    print("Testing sliding window chunker:")
    print("=" * 80)
    
    chunks = sliding_window_chunks(sample_text, window_size=200, overlap=50)
    
    print(f"\nGenerated {len(chunks)} overlapping chunks:")
    for i, (chunk, start, end) in enumerate(chunks, 1):
        print(f"\nChunk {i} (pos {start}-{end}):")
        print(f"  {chunk[:100]}...")
    
    print("\n" + "=" * 80)
    print("Testing deduplication:")
    print("=" * 80)
    
    # Simulate scored chunks with overlap
    scored = [
        (chunks[0][0], 0.8, chunks[0][1], chunks[0][2]),
        (chunks[1][0], 0.6, chunks[1][1], chunks[1][2]),
        (chunks[2][0], 0.3, chunks[2][1], chunks[2][2]),  # Below threshold
    ]
    
    deduplicated = deduplicate_overlapping_chunks(scored, score_threshold=0.5)
    
    print(f"\nAfter deduplication (threshold=0.5): {len(deduplicated)} chunks kept")
    for i, (text, score) in enumerate(deduplicated, 1):
        print(f"\n[{i}] Score: {score:.2f}")
        print(f"    {text[:100]}...")
