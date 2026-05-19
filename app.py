"""
ISOM5240 Group Project — VibeSound
Optimised for Streamlit Cloud free tier (1 GB RAM)
- Image captioning: ViT-GPT2 (small)
- Mood classifier: DistilRoBERTa emotion (small)
- Prompt builder: flan-t5-small (8‑bit quantised)
- Music generation: HF Inference API (no local model)
"""
import os
import time
import tempfile
import numpy as np
import streamlit as st
import requests
import torch
from PIL import Image
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from huggingface_hub import InferenceClient

# ============================================================
# CONFIGURATION
# ============================================================
# Smaller captioning model
CAPTION_MODEL = "ydshieh/vit-gpt2-coco-en"

# Smaller emotion classifier (distilroberta-base)
MOOD_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Prompt builder (flan-t5-small) – will be 8‑bit quantised
PROMPT_MODEL = "google/flan-t5-small"

# MusicGen via API
MUSICGEN_MODEL = "facebook/musicgen-small"

# Mood labels from the new model
MOOD_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral",
    "sadness", "surprise"
]
# Remap to your original 6 music moods
MOOD_REMAP = {
    "joy": "happy", "sadness": "sad", "anger": "intense",
    "disgust": "intense", "fear": "intense", "surprise": "surprised",
    "neutral": "neutral"
}

MOOD_FALLBACK = {
    "happy": "upbeat acoustic guitar, bright cheerful, fast tempo",
    "sad": "slow melancholic piano, minor key, cinematic",
    "romantic": "soft romantic acoustic guitar, warm gentle",
    "intense": "dramatic orchestra, heavy drums, fast, dark",
    "surprised": "playful quirky ukulele, bouncy, dynamic",
    "neutral": "calm ambient background, soft instrumental",
}

MOOD_EMOJI = {
    "happy": "😊", "sad": "😢", "romantic": "❤️",
    "intense": "😠", "surprised": "😲", "neutral": "😐",
}
MOOD_COLOR = {
    "happy": "#FFD700", "sad": "#4A90D9", "romantic": "#E91E8C",
    "intense": "#E74C3C", "surprised": "#FF6B35", "neutral": "#888888",
}

# ============================================================
# MEMORY OPTIMISATIONS
# ============================================================
torch.set_num_threads(1)          # reduce CPU threads
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# MODEL LOADING (all local, except MusicGen)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_captioner():
    """ViT-GPT2 captioning – ~600MB"""
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    processor = ViTImageProcessor.from_pretrained(CAPTION_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL)
    model.eval()
    return processor, model, tokenizer

@st.cache_resource(show_spinner=False)
def load_mood_classifier():
    """DistilRoBERTa emotion – ~300MB"""
    return pipeline(
        "text-classification",
        model=MOOD_MODEL,
        top_k=None,
        device=-1,  # CPU
    )

@st.cache_resource(show_spinner=False)
def load_prompt_builder():
    """flan-t5-small in 8‑bit – ~150MB"""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        PROMPT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model

def generate_music_api(prompt: str, token: str) -> bytes:
    """Call HF Inference API for MusicGen"""
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{MUSICGEN_MODEL}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.content

def normalise_mood(label: str) -> str:
    label = label.lower()
    return MOOD_REMAP.get(label, "neutral")

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="VibeSound – Light", page_icon="🎵", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0D0D0D, #1A0A2E); }
    .title-text { font-size: 2.5rem; font-weight: 700;
                  background: linear-gradient(90deg, #E91E8C, #FF6B35);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .caption-box, .prompt-box { background: rgba(255,255,255,0.05); padding: 12px;
                                 border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🎵 VibeSound (Light)</p>', unsafe_allow_html=True)
st.markdown("Optimised for 1 GB RAM – all pipelines local, MusicGen via API.")

col_left, col_right = st.columns(2)
with col_left:
    uploaded = st.file_uploader("📸 Upload reel photo", type=["jpg","png","jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
with col_right:
    user_text = st.text_area("💬 Describe your vibe (optional)", height=120,
                             placeholder="e.g. missing summer... best day ever...")
    st.caption("Leave empty → neutral mood")

if st.button("🎵 Generate Music", type="primary", disabled=(uploaded is None), use_container_width=True):
    if uploaded is None:
        st.stop()

    # ---------- 1. Captioning ----------
    st.markdown("---")
    st.markdown("**📝 1. Captioning photo (ViT-GPT2)**")
    with st.spinner("..."):
        try:
            processor, caption_model, tokenizer = load_captioner()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            caption = "a scenic photo"
            st.warning(f"Caption fallback: {e}")
    st.markdown(f'<div class="caption-box">📷 {caption}</div>', unsafe_allow_html=True)

    # ---------- 2. Mood detection ----------
    st.markdown("**🎭 2. Detecting mood (DistilRoBERTa)**")
    with st.spinner("..."):
        try:
            if user_text.strip():
                classifier = load_mood_classifier()
                raw = classifier(user_text.strip())[0]
                raw.sort(key=lambda x: x["score"], reverse=True)
                top_label = raw[0]["label"]
                top_score = raw[0]["score"]
                top_mood = normalise_mood(top_label)
            else:
                top_mood, top_score = "neutral", 1.0
        except Exception as e:
            st.error(f"Mood error: {e}")
            st.stop()

    emoji, color = MOOD_EMOJI.get(top_mood, "🎶"), MOOD_COLOR.get(top_mood, "#888")
    st.markdown(f'<div style="background:{color}; display:inline-block; padding:5px 15px; border-radius:30px;">{emoji} {top_mood.capitalize()} ({top_score*100:.1f}%)</div>', unsafe_allow_html=True)

    # ---------- 3. Prompt building (8‑bit flan-t5) ----------
    st.markdown("**✍️ 3. Building music prompt (flan-t5-small 8‑bit)**")
    with st.spinner("..."):
        try:
            tokenizer, prompt_model = load_prompt_builder()
            instruction = (
                f"Generate background music keywords for an Instagram reel. "
                f"Scene: {caption}. Mood: {top_mood}. "
                f"Output comma-separated keywords only, max 15 words."
            )
            inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=128)
            outputs = prompt_model.generate(**inputs, max_new_tokens=40)
            music_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not music_prompt or len(music_prompt.split()) > 20:
                music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")
        except Exception as e:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")
            st.warning(f"Prompt fallback: {e}")
    st.markdown(f'<div class="prompt-box">🎼 {music_prompt}</div>', unsafe_allow_html=True)

    # ---------- 4. Music generation via API ----------
    st.markdown("**🎶 4. Generating music (HF API – MusicGen-small)**")
    if "HF_TOKEN" not in st.secrets:
        st.error("Missing `HF_TOKEN` in secrets. Please add it.")
        st.stop()

    with st.spinner("Composing (20‑30 sec)..."):
        try:
            audio_bytes = generate_music_api(music_prompt, st.secrets["HF_TOKEN"])
        except Exception as e:
            st.error(f"MusicGen API failed: {e}")
            st.stop()

    st.success("✅ Music ready!")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button("⬇️ Download WAV", data=audio_bytes, file_name=f"vibesound_{top_mood}.wav", mime="audio/wav")

    # Summary
    st.markdown("---")
    st.markdown("### 📊 Summary")
    st.markdown(f"**Caption:** {caption}")
    st.markdown(f"**Mood:** {emoji} {top_mood.capitalize()} ({top_score*100:.1f}%)")
    st.markdown(f"**Prompt:** {music_prompt}")

# Sidebar info
with st.sidebar:
    st.markdown("### 🔧 Optimisations")
    st.success("✅ ViT-GPT2 captioning (~600MB)")
    st.success("✅ DistilRoBERTa emotion (~300MB)")
    st.success("✅ flan-t5-small 8‑bit (~150MB)")
    st.info("🎵 MusicGen via HF API (0MB local)")
    st.markdown(f"**Total local RAM ≈ 1.05 GB** – fits free tier")
    st.markdown("---")
    st.caption("Add your `HF_TOKEN` in Streamlit secrets for MusicGen API.")
