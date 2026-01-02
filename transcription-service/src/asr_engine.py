import torch
import torchaudio
import librosa
import numpy as np
import warnings
import os
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoProcessor,
    SeamlessM4Tv2ForSpeechToText
)
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CodeSwitchASR:
    def __init__(self):
        print(f"LOADING MODELS on {DEVICE}")
        print("Loading LID (VoxLingua107)...")
        self.lid_model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa", 
            run_opts={"device": DEVICE},
            savedir="tmp_lid"
        )
        
        print("Loading Whisper (Telugu Context)...")
        self.wh_pid = "vasista22/whisper-telugu-medium"
        self.wh_proc = WhisperProcessor.from_pretrained(self.wh_pid)
        self.wh_model = WhisperForConditionalGeneration.from_pretrained(self.wh_pid).to(DEVICE)
        
        print("Loading Wav2Vec (Telugu Phonetics)...")
        self.w2v_pid = "anuragshas/wav2vec2-large-xlsr-53-telugu" 
        self.w2v_proc = Wav2Vec2Processor.from_pretrained(self.w2v_pid)
        self.w2v_model = Wav2Vec2ForCTC.from_pretrained(self.w2v_pid).to(DEVICE)
        
        print("Loading SeamlessM4T (English)...")
        self.sm_pid = "facebook/seamless-m4t-v2-large"
        self.sm_proc = AutoProcessor.from_pretrained(self.sm_pid)
        self.sm_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.sm_pid, torch_dtype=torch.float16).to(DEVICE)
        
        print("All models loaded successfully.")
        
    def rover_vote(self,hyp_whisper,conf_whisper,hyp_w2v,conf_w2v,w2v_weight=0.3):
        score_w=conf_whisper * (1 - w2v_weight)
        score_v=conf_w2v * w2v_weight
        if hyp_whisper.strip() == hyp_w2v.strip(): return hyp_whisper
        if len(hyp_w2v) < 2 and len(hyp_whisper) > 2: return hyp_whisper
        return hyp_w2v if score_v > score_w else hyp_whisper
    
    def run_lid_scan(self,audio_path,window=5,step=0.3):
        if not os.path.exists(audio_path):
            return []
        wav,fs=torchaudio.load(audio_path)
        if wav.dim()==2 and wav.size(0)>1:
            wav=torch.mean(wav,dim=0,keepdim=True)
        if fs!=16000:
            resampler=torchaudio.transforms.Resample(fs,16000)
            wav=resampler(wav)
        wav=wav.squeeze().to(DEVICE)
        duration=wav.shape[0]/16000
        if duration < window:
            return [{"start":0.0,"end":duration,"lang":"unknown"}]
        timeline=[]
        for t in np.arange(0,duration-window,step):
            start=int(t*16000)
            end=int((t+window)*16000)
            segment=wav[start:end].unsqueeze(0)
            energy=torch.mean(torch.abs(segment)).item()
            if energy<0.005:
                continue
            prob,_,_,label=self.lid_model.classify_batch(segment)
            lang=label[0]
            final_lang = lang if lang in ["te: Telugu", "en"] else "en"
            timeline.append({"t": t, "lang": final_lang})
        if not timeline:
            return []
        segments=[]
        curr_lang=timeline[0]["lang"]
        curr_start=timeline[0]["t"]
        for i in range(1,len(timeline)):
            if timeline[i]["lang"] != curr_lang:
                segments.append({"start":curr_start,"end":timeline[i]["t"]+step,"lang":curr_lang})
                curr_lang=timeline[i]["lang"]
                curr_start=timeline[i]["t"]
        segments.append({"start":curr_start,"end":duration,"lang":curr_lang})
        return segments
    
    def transcribe_chunk_telugu(self,audio_chunk):
        inputs=self.wh_proc(audio_chunk,sampling_rate=16000,return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out=self.wh_model.generate(inputs.input_features,num_beams=4,return_dict_in_generate=True,output_scores=True)
        wh_text=self.wh_proc.batch_decode(out.sequences,skip_special_tokens=True)[0]
        wh_conf = torch.exp(out.sequences_scores).mean().item() if out.sequences_scores is not None else 0.8
        inputs_w2v=self.w2v_proc(audio_chunk,sampling_rate=16000,return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out=self.w2v_model(inputs_w2v.input_values).logits
        pred_ids=torch.argmax(out,dim=-1)[0]
        w2v_text=self.w2v_proc.decode(pred_ids)
        w2v_conf = torch.max(torch.softmax(out, dim=-1), dim=-1).values.mean().item()
        
        final_text=self.rover_vote(wh_text,wh_conf,w2v_text,w2v_conf)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return final_text.strip()
    
    def transcribe_chunk_english(self, audio_chunk):
        inputs = self.sm_proc(audio=audio_chunk, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.sm_model.generate(**inputs, tgt_lang="eng", num_beams=3)
        text = self.sm_proc.decode(out[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return text.strip()
    
    def process_file(self,audio_path):
        segments=self.run_lid_scan(audio_path)
        full_audio,_=librosa.load(audio_path,sr=16000)
        final_transcript=[]
        print(f"Processing {len(segments)} segments...")
        
        for seg in segments:
            s_samp = int(seg['start'] * 16000)
            e_samp = int(seg['end'] * 16000)
            chunk = full_audio[s_samp:e_samp]
            
            if len(chunk) < 1600: continue
            
            if seg['lang'] == 'te: Telugu':
                text = self.transcribe_chunk_telugu(chunk)
            else:
                text = self.transcribe_chunk_english(chunk)
            
            final_transcript.append(text)

        return " ".join(final_transcript)
        
        