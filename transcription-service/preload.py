import os

if not os.path.exists("tmp_lid"):
    os.makedirs("tmp_lid")

print("Starting model download for baking into image...")

try:
    from speechbrain.inference.classifiers import EncoderClassifier
    print("Downloading SpeechBrain LID...")
    EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="tmp_lid"
    )
except ImportError:
    print("SpeechBrain not found or error.")
except Exception as e:
    print(f"Error downloading SpeechBrain: {e}")

try:
    from transformers import (
        Wav2Vec2ForCTC, Wav2Vec2Processor,
        WhisperForConditionalGeneration, WhisperProcessor,
        AutoProcessor, SeamlessM4Tv2ForSpeechToText
    )
    
    print("Downloading Whisper (vasista22/whisper-telugu-medium)...")
    wh_pid = "vasista22/whisper-telugu-medium"
    WhisperProcessor.from_pretrained(wh_pid)
    WhisperForConditionalGeneration.from_pretrained(wh_pid)
    
    print("Downloading Wav2Vec2 (anuragshas/wav2vec2-large-xlsr-53-telugu)...")
    w2v_pid = "anuragshas/wav2vec2-large-xlsr-53-telugu"
    Wav2Vec2Processor.from_pretrained(w2v_pid)
    Wav2Vec2ForCTC.from_pretrained(w2v_pid)

    print("Downloading SeamlessM4T (facebook/seamless-m4t-v2-large)...")
    sm_pid = "facebook/seamless-m4t-v2-large"
    AutoProcessor.from_pretrained(sm_pid)

    SeamlessM4Tv2ForSpeechToText.from_pretrained(sm_pid)

except ImportError:
    print("Transformers not found.")
except Exception as e:
    print(f"Error downloading Hugging Face models: {e}")

print("Preload complete.")
