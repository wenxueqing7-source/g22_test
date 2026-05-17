"""
ISOM5240 Group Project — VibeSound
Background Music Generator for Instagram Reels

HuggingFace Pipelines (transformers.pipeline):
  ★ Pipeline 1: image-to-text         (Salesforce/blip-image-captioning-base, HF Inference API)
  ★ Pipeline 2: text-classification   (distilbert-base-uncased, FINE-TUNED on go_emotions remapped)
  ★ Pipeline 3: text2text-generation  (google/flan-t5-small, pre-trained, local)

Music Generation (HF Inference API — not a pipeline):
  facebook/musicgen-small
"""

import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image
import io

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
FINETUNED_MODEL = "YOUR_HF_USERNAME/distilbert-go-emotions-music"  # ← replace after uploading
BLIP_MODEL      = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL    = "google/flan-t5-small"
MUSICGEN_MODEL  = "facebook/musicgen-small"

MOOD_FALLBACK = {
    "happy":     "upbeat acoustic guitar, bright cheerful, fast tempo, major key",
    "sad":       "slow melancholic piano, minor key, cinematic, emotional",
    "romantic":  "soft romantic acoustic guitar, warm gentle, slow tempo",
    "intense":   "dramatic orchestra, heavy drums, fast, dark, powerful",
    "surprised": "playful quirky ukulele, bouncy, dynamic, bright",
    "neutral":   "calm ambient background, soft instrumental, peaceful",
}

MOOD_EMOJI = {
    "happy": "😊", "sad": "😢", "romantic": "❤️",
    "intense": "😠", "surprised": "😲", "neutral": "😐",
}

MOOD_COLOR = {
    "happy": "#FFD700", "sad": "#4A90D9", "romantic": "#E91E8C",
    "intense": "#E74C3C", "surprised": "#FF6B35", "neutral": "#888888",
}

# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator",
    page_icon="🎵",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0D0D0D 0%, #1A0A2E 100%); }
    .title-text { font-size: 2.6rem; font-weight: 700;
                  background: linear-gradient(90deg, #E91E8C, #FF6B35);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: #aaa; font-size: 1rem; margin-top: -8px; }
    .card { background: rgba(255,255,255,0.05); border-radius: 16px;
            padding: 20px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
    .step-label { color: #888; font-size: 0.8rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .caption-box { color: #eee; font-style: italic; font-size: 1rem; padding: 12px;
                   background: rgba(255,255,255,0.04);
                   border-left: 3px solid #E91E8C; border-radius: 4px; }
    .prompt-box { color: #eee; font-size: 0.95rem; padding: 12px;
                  background: rgba(255,255,255,0.04);
                  border-left: 3px solid #FF6B35; border-radius: 4px; }
    .mood-badge { display: inline-block; padding: 6px 18px; border-radius: 30px;
                  font-weight: 600; font-size: 1rem; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate background music for your Instagram Reel</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
#  MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    """
    ★ PIPELINE 2: text-classification [FINE-TUNED]
    Model : distilbert-base-uncased fine-tuned on go_emotions (28 labels → 6 music moods)
    Task  : User free-text → mood label (happy/sad/romantic/intense/surprised/neutral)
    RAM   : ~268MB (safe for Streamlit Cloud free tier)
    """
    return pipeline(
        "text-classification",
        model=FINETUNED_MODEL,
        return_all_scores=True,
    )

@st.cache_resource(show_spinner=False)
def load_prompt_builder():
    """
    ★ PIPELINE 3: text2text-generation (pre-trained, no fine-tuning needed)
    Model : google/flan-t5-small (instruction-tuned by Google)
    Task  : (caption + mood) → comma-separated music style keywords
    RAM   : ~308MB (safe for Streamlit Cloud free tier)
    """
    return pipeline(
        "text2text-generation",
        model=FLANT5_MODEL,
        max_new_tokens=40,
    )

def get_hf_client():
    """HF Inference API client for BLIP (Pipeline 1) and MusicGen."""
    token = st.secrets.get("HF_TOKEN", "")
    return InferenceClient(token=token) if token else None

# ─────────────────────────────────────────────
#  PARALLEL INPUT — Photo + Text side by side
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📸 Upload your reel photo")
    uploaded = st.file_uploader(
        "Photo required",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with col_right:
    st.markdown("#### 💬 How are you feeling?")
    user_text = st.text_area(
        "Optional",
        placeholder="e.g. missing summer so much...\nbest day ever with my girls\ncan't believe this view 😍",
        height=160,
        label_visibility="collapsed",
    )
    st.caption("Optional — leave empty to default to neutral mood")

st.markdown("---")

# Generate button — only active when photo uploaded
generate_btn = st.button(
    "🎵 Generate My Background Music",
    type="primary",
    disabled=(uploaded is None),
    use_container_width=True,
)

if uploaded is None:
    st.caption("⬆️ Upload a photo to activate the generate button")

# ─────────────────────────────────────────────
#  GENERATION PIPELINE
# ─────────────────────────────────────────────
if generate_btn and uploaded:

    client = get_hf_client()
    if client is None:
        st.error("⚠️ HF_TOKEN not found. Add it under Streamlit Cloud → Settings → Secrets.")
        st.stop()

    # ── PIPELINE 1: BLIP Image Captioning (HF Inference API) ──────────
    st.markdown("---")
    st.markdown('<p class="step-label">★ Pipeline 1 — Image-to-Text (BLIP)</p>',
                unsafe_allow_html=True)

    with st.spinner("Reading your photo scene..."):
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            caption_result = client.image_to_text(
                image=img_bytes.read(),
                model=BLIP_MODEL,
            )
            caption = caption_result if isinstance(caption_result, str) \
                      else caption_result[0].get("generated_text", "a scenic photo")
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            st.stop()

    st.markdown(f'<div class="caption-box">📝 Scene: {caption}</div>',
                unsafe_allow_html=True)

    # ── PIPELINE 2: Emotion Classification (local fine-tuned DistilBERT)
    st.markdown(" ")
    st.markdown('<p class="step-label">★ Pipeline 2 — Text Classification (DistilBERT fine-tuned)</p>',
                unsafe_allow_html=True)

    with st.spinner("Detecting mood from your text..."):
        try:
            if user_text.strip():
                classifier = load_emotion_classifier()
                scores     = classifier(user_text.strip())[0]
                scores     = sorted(scores, key=lambda x: x["score"], reverse=True)
                top_mood   = scores[0]["label"].lower()
                top_score  = scores[0]["score"]
            else:
                # No text input → default neutral
                top_mood   = "neutral"
                top_score  = 1.0
                scores     = [{"label": "neutral", "score": 1.0}]
        except Exception as e:
            st.error(f"Mood detection failed: {e}")
            st.stop()

    emoji = MOOD_EMOJI.get(top_mood, "🎶")
    color = MOOD_COLOR.get(top_mood, "#888")

    col_mood, col_chart = st.columns([1, 2])
    with col_mood:
        st.markdown(
            f'<div class="mood-badge" style="background:{color};">'
            f'{emoji} {top_mood.capitalize()}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {top_score*100:.1f}%")
        if not user_text.strip():
            st.caption("*(no text → defaulted to neutral)*")
    with col_chart:
        if user_text.strip() and len(scores) >= 3:
            top3 = {s["label"].capitalize(): round(s["score"], 3) for s in scores[:3]}
            st.bar_chart(top3, height=100)

    # ── PIPELINE 3: Prompt Construction (local flan-t5-small) ─────────
    st.markdown(" ")
    st.markdown('<p class="step-label">★ Pipeline 3 — Music Prompt Builder (flan-t5-small)</p>',
                unsafe_allow_html=True)

    with st.spinner("Building music prompt..."):
        try:
            prompt_builder = load_prompt_builder()
            instruction = (
                f"Generate background music keywords for an Instagram reel. "
                f"Scene: {caption}. "
                f"Mood: {top_mood}. "
                f"Output comma-separated music style keywords only, maximum 15 words."
            )
            raw_output   = prompt_builder(instruction)[0]["generated_text"].strip()
            music_prompt = raw_output if raw_output and len(raw_output.split()) <= 20 \
                           else MOOD_FALLBACK.get(top_mood, "calm ambient music")
        except Exception:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")

    st.markdown(f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
                unsafe_allow_html=True)

    # ── MUSIC GENERATION: MusicGen (HF Inference API) ─────────────────
    st.markdown(" ")
    st.markdown('<p class="step-label">Music Generation — MusicGen-small (HF Inference API)</p>',
                unsafe_allow_html=True)

    with st.spinner("Composing your music clip (20–30 sec)..."):
        try:
            audio_bytes = client.text_to_audio(
                text=music_prompt,
                model=MUSICGEN_MODEL,
            )
        except Exception as e:
            st.error(f"Music generation failed: {e}")
            st.stop()

    st.success("✅ Your background music is ready!")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        label="⬇️ Download Music (.wav)",
        data=audio_bytes,
        file_name=f"vibesound_{top_mood}.wav",
        mime="audio/wav",
    )

    # ── RESULT SUMMARY ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Generation Summary")
    data = {
        "Scene caption":  caption,
        "Your text":      user_text.strip() if user_text.strip() else "*(none)*",
        "Detected mood":  f"{emoji} {top_mood.capitalize()} ({top_score*100:.1f}%)",
        "Music prompt":   music_prompt,
    }
    for k, v in data.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Pipeline Architecture")
    st.success("✅ **Pipeline 1** `image-to-text`\n\nBLIP-base (pre-trained)\n\nPhoto → scene caption\n\n*via HF Inference API*")
    st.success("✅ **Pipeline 2** `text-classification`\n\nDistilBERT ⭐ fine-tuned\n\nUser text → mood label\n\n*local ~268MB*")
    st.success("✅ **Pipeline 3** `text2text-generation`\n\nflan-t5-small (pre-trained)\n\nCaption + mood → prompt\n\n*local ~308MB*")
    st.info("🎵 **MusicGen-small**\n\nHF Inference API\n\nPrompt → audio\n\n*not a pipeline*")
    st.markdown("---")
    st.markdown("### 🎭 Mood Classes")
    for mood, emoji in MOOD_EMOJI.items():
        st.markdown(f"{emoji} **{mood.capitalize()}**")
    st.markdown("---")
    st.caption("ISOM5240 Group Project · HuggingFace 🤗")
