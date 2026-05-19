"""
ISOM5240 Group Project — VibeSound
Background Music Generator for Instagram Reels

HuggingFace Pipelines (transformers.pipeline):
  ★ Pipeline 1: image-to-text         (local BLIP float16)
  ★ Pipeline 2: text-classification   (local DistilBERT)
  ★ Pipeline 3: text2text-generation  (local flan-t5-small)

Music Generation: HF Inference API (facebook/musicgen-small)
"""
import streamlit as st
import requests
from PIL import Image
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import torch

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
USE_FINETUNED_MODEL = False
PLACEHOLDER_MODEL   = "bhadresh-savani/distilbert-base-uncased-emotion"
FINETUNED_MODEL     = "MelodyWEN7/vibesound-music-mood-classifier"

BLIP_MODEL          = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL        = "google/flan-t5-small"
MUSICGEN_MODEL      = "facebook/musicgen-small"

PLACEHOLDER_REMAP = {
    "joy": "happy", "sadness": "sad", "love": "romantic",
    "anger": "intense", "fear": "intense", "surprise": "surprised",
}
FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

MOOD_FALLBACK = {
    "happy": "upbeat acoustic guitar, bright cheerful, fast tempo, major key",
    "sad": "slow melancholic piano, minor key, cinematic, emotional",
    "romantic": "soft romantic acoustic guitar, warm gentle, slow tempo",
    "intense": "dramatic orchestra, heavy drums, fast, dark, powerful",
    "surprised": "playful quirky ukulele, bouncy, dynamic, bright",
    "neutral": "calm ambient background, soft instrumental, peaceful",
}

MOOD_EMOJI = {"happy":"😊","sad":"😢","romantic":"❤️","intense":"😠","surprised":"😲","neutral":"😐"}
MOOD_COLOR = {"happy":"#FFD700","sad":"#4A90D9","romantic":"#E91E8C","intense":"#E74C3C","surprised":"#FF6B35","neutral":"#888888"}

# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="VibeSound — Reel Music Generator", page_icon="🎵", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0D0D0D 0%, #1A0A2E 100%); }
    .title-text { font-size: 2.6rem; font-weight: 700;
                  background: linear-gradient(90deg, #E91E8C, #FF6B35);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: #aaa; font-size: 1rem; margin-top: -8px; }
    .caption-box { background: rgba(255,255,255,0.04); border-left: 3px solid #E91E8C; padding: 12px; border-radius: 4px; color: #eee; }
    .prompt-box { background: rgba(255,255,255,0.04); border-left: 3px solid #FF6B35; padding: 12px; border-radius: 4px; color: #eee; }
    .mood-badge { display: inline-block; padding: 6px 18px; border-radius: 30px; font-weight: 600; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate background music for your Instagram Reel</p>', unsafe_allow_html=True)
if not USE_FINETUNED_MODEL:
    st.warning("⚠️ Running with placeholder emotion model. Set `USE_FINETUNED_MODEL = True` after fine-tuning.", icon="⚠️")
st.markdown("---")

# ─────────────────────────────────────────────
#  MODEL LOADING (cached, local)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL, torch_dtype=torch.float16)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    model_name = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    return pipeline("text-classification", model=model_name, top_k=None)

@st.cache_resource(show_spinner=False)
def load_prompt_builder():
    return pipeline("text2text-generation", model=FLANT5_MODEL, max_new_tokens=40)

def normalise_mood(label: str) -> str:
    label = label.lower()
    if USE_FINETUNED_MODEL:
        return label if label in FINETUNED_LABELS else "neutral"
    return PLACEHOLDER_REMAP.get(label, "neutral")

# ── HF API music generation ──────────────────
def generate_music_api(prompt: str, token: str) -> bytes:
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{MUSICGEN_MODEL}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.content

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎵 Music Generation")
    st.info("Using **HF Inference API** (no local MusicGen) – works within free tier RAM limits.")
    if "HF_TOKEN" not in st.secrets:
        st.error("❌ `HF_TOKEN` not found in Streamlit secrets. Add it to use music generation.")
    else:
        st.success("✅ HF token loaded – music generation ready.")

    st.markdown("---")
    st.markdown("### 🔬 Pipelines (local)")
    st.success("✅ BLIP (image→caption) – float16")
    st.success("✅ DistilBERT (caption→mood)")
    st.success("✅ flan-t5-small (mood→prompt)")
    st.markdown("---")
    st.markdown("### 🎭 Mood Classes")
    for mood, em in MOOD_EMOJI.items():
        st.markdown(f"{em} **{mood.capitalize()}**")
    st.caption("ISOM5240 Group Project · HuggingFace 🤗")

# ─────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📸 Upload your reel photo")
    uploaded = st.file_uploader("Photo required", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with col_right:
    st.markdown("#### 💬 Describe your vibe (optional)")
    user_text = st.text_area("", placeholder="e.g. missing summer so much...\nbest day ever with my girls", height=160, label_visibility="collapsed")
    st.caption("Leave empty → neutral mood")

st.markdown("---")
generate_btn = st.button("🎵 Generate Background Music", type="primary", disabled=(uploaded is None), use_container_width=True)
if uploaded is None:
    st.caption("⬆️ Upload a photo to activate")

# ─────────────────────────────────────────────
#  GENERATION PIPELINE
# ─────────────────────────────────────────────
if generate_btn and uploaded:
    # Pipeline 1: BLIP captioning
    st.markdown("---")
    st.markdown('<p class="step-label">★ Pipeline 1 — image captioning (BLIP)</p>', unsafe_allow_html=True)
    with st.spinner("Reading your photo..."):
        try:
            processor, blip_model = load_blip()
            inputs = processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Captioning failed: {e}")
            caption = "a scenic photo"
            st.warning("Using fallback caption.")
    st.markdown(f'<div class="caption-box">📝 Scene: {caption}</div>', unsafe_allow_html=True)

    # Pipeline 2: Emotion classification
    st.markdown(" ")
    st.markdown('<p class="step-label">★ Pipeline 2 — mood detection</p>', unsafe_allow_html=True)
    with st.spinner("Detecting mood..."):
        try:
            if user_text.strip():
                classifier = load_emotion_classifier()
                raw_scores = classifier(user_text.strip())[0]
                raw_scores.sort(key=lambda x: x["score"], reverse=True)
                top_mood = normalise_mood(raw_scores[0]["label"])
                top_score = raw_scores[0]["score"]
                from collections import defaultdict
                merged = defaultdict(float)
                for s in raw_scores:
                    merged[normalise_mood(s["label"])] += s["score"]
                display_scores = sorted(merged.items(), key=lambda x: -x[1])
            else:
                top_mood, top_score = "neutral", 1.0
                display_scores = [("neutral", 1.0)]
        except Exception as e:
            st.error(f"Mood detection failed: {e}")
            st.stop()

    emoji, color = MOOD_EMOJI.get(top_mood, "🎶"), MOOD_COLOR.get(top_mood, "#888")
    col_mood, col_chart = st.columns([1,2])
    with col_mood:
        st.markdown(f'<div class="mood-badge" style="background:{color};">{emoji} {top_mood.capitalize()}</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {top_score*100:.1f}%")
        if not user_text.strip():
            st.caption("*(no text → neutral)*")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v,3) for k,v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    # Pipeline 3: Prompt building with flan-t5
    st.markdown(" ")
    st.markdown('<p class="step-label">★ Pipeline 3 — music prompt (flan-t5)</p>', unsafe_allow_html=True)
    with st.spinner("Crafting music prompt..."):
        try:
            prompt_builder = load_prompt_builder()
            instruction = f"Generate background music keywords for an Instagram reel. Scene: {caption}. Mood: {top_mood}. Output comma-separated keywords only, max 15 words."
            raw_output = prompt_builder(instruction)[0]["generated_text"].strip()
            music_prompt = raw_output if raw_output and len(raw_output.split()) <= 20 else MOOD_FALLBACK.get(top_mood, "calm ambient music")
        except Exception:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")
    st.markdown(f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>', unsafe_allow_html=True)

    # Music Generation via HF API (no local model)
    st.markdown(" ")
    st.markdown('<p class="step-label">🎵 Music Generation (HF Inference API)</p>', unsafe_allow_html=True)
    if "HF_TOKEN" not in st.secrets:
        st.error("Cannot generate music: `HF_TOKEN` is missing from Streamlit secrets. Please add it.")
        st.stop()

    with st.spinner("Composing your music clip (20–30 seconds)..."):
        try:
            audio_bytes = generate_music_api(music_prompt, st.secrets["HF_TOKEN"])
        except Exception as e:
            st.error(f"Music generation failed: {e}")
            st.info("Make sure your Hugging Face token is valid and has access to facebook/musicgen-small.")
            st.stop()

    st.success("✅ Music generated!")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(label="⬇️ Download WAV", data=audio_bytes, file_name=f"vibesound_{top_mood}.wav", mime="audio/wav", use_container_width=True)

    # Summary
    st.markdown("---")
    st.markdown("### 📊 Generation Summary")
    st.markdown(f"**Scene:** {caption}")
    st.markdown(f"**Your text:** {user_text.strip() if user_text.strip() else '*(none)*'}")
    st.markdown(f"**Detected mood:** {emoji} {top_mood.capitalize()} ({top_score*100:.1f}%)")
    st.markdown(f"**Music prompt:** {music_prompt}")
