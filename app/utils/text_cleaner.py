import re
from langchain_core.runnables import RunnableLambda

def clean_scraped_text(text):
    if not text: return []
    
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    text = re.sub(r'\[.*?\]', '', text)
    
    text = re.sub(r'File:.*|Image:.*', '', text)

    lines = text.split('\n')
    cleaned_lines = []

    noise_keywords = [
        "login", "register", "sign in", "sign up", "subscribe", 
        "breadcrumb", "share this", "follow us", "related crops", 
        "download as", "printable version", "terms & condition",
        "privacy policy", "disclaimer", "copyright", "all rights reserved",
        "contact us", "about us", "agro services", "bid for product",
        "delete your account", "confirmation"
    ]

    for line in lines:
        line = line.strip()
        
        if not line: continue
        
        if not any(c.isalnum() for c in line):
            continue

        if any(keyword in line.lower() for keyword in noise_keywords):
            continue
            
        if len(line) < 5 and not line.startswith('#'):
            continue

        cleaned_lines.append(line)
        
    return cleaned_lines

def merge_paragraphs(lines):
    merged_paragraphs = []
    current_para = []

    for line in lines:
        line = line.strip()
        
        is_header = line.startswith('#')
        is_bullet = line.startswith('*') or line.startswith('-') or line.startswith('+')
        
        if is_header or is_bullet:
            if current_para:
                merged_paragraphs.append(" ".join(current_para))
                current_para = []
            merged_paragraphs.append(line)
        else:
            if current_para:
                current_para.append(line)
            else:
                current_para = [line]
            
    if current_para:
        merged_paragraphs.append(" ".join(current_para))
        
    return merged_paragraphs

def create_smart_chunks(paragraphs, max_chars=1500, min_chars=100):
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
                current_chunk = para 
            else:
                current_chunk += "\n\n" + para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

clean_runnable = RunnableLambda(clean_scraped_text)
merge_ruunable = RunnableLambda(merge_paragraphs)
chunk_runnable = RunnableLambda(create_smart_chunks)

processing_chain= clean_runnable | merge_ruunable | chunk_runnable




# clean_lines = clean_scraped_text(raw_input)

# coherent_paragraphs = merge_paragraphs(clean_lines)

# final_chunks = create_smart_chunks(coherent_paragraphs)

# print(f"Total Chunks: {len(final_chunks)}\n")
# for i, chunk in enumerate(final_chunks):
#     print(f"--- CHUNK {i+1} ---\n{chunk}\n")
