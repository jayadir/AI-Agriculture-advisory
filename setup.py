from huggingface_hub import snapshot_download
import time

print("🚀 Starting Model Download (Safe Mode)...")
print("📦 Model: jinaai/jina-embeddings-v3")
print("⏳ This involves downloading ~2.3GB. Please wait...")

start_time = time.time()

# We use snapshot_download to fetch the files without executing the custom code yet.
# This prevents the Windows 'Flash Attention' crash during the setup phase.
snapshot_download(
    repo_id="jinaai/jina-embeddings-v3",
    local_files_only=False,
    revision="main" 
)

end_time = time.time()
print(f"\n✅ Download Complete! Time taken: {end_time - start_time:.2f}s")
print("🎉 You can now run 'uvicorn app.main:app --reload'")