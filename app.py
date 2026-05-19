"""
ISOM5240 Group Project — VibeSound
Background Music Generator for Instagram Reels

HuggingFace Pipelines (transformers.pipeline):
  ★ Pipeline 1: image-to-text         (Salesforce/blip-image-captioning-base, LOCAL float16)
  ★ Pipeline 2: text-classification   (distilbert-base-uncased, FINE-TUNED on go_emotions)
  ★ Pipeline 3: text2text-generation  (google/flan-t5-small, pre-trained)

Music Generation (HF Inference API — not a pipeline):
  facebook/musicgen-small
"""

import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
from PIL import Image
import torch
import io

# ─────────────────────────────────────────────
#  CONFIG
#  ⚠️  Before deploying:
#  1. Run the fine-tuning notebook
#  2. Set USE_FINETUNED_MODEL = True
#  3. Replace FINETUNED_MODEL with your actual HF model path
# ─────────────────────────────────────────────
USE_FINETUNED_MODEL = False   # ← set True after fine-tuning is done

# Placeholder: real existing model used until fine-tuning is complete
PLACEHOLDER_MODEL   = "bhadresh-savani/distilbert-base-uncased-emotion"
# Replace this with your actual fine-tuned model after running the notebook:
FINETUNED_MODEL     = "MelodyWEN7/vibesound-music-mood-classifier"

BLIP_MODEL          = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL        = "google/flan-t5-small"
MUSICGEN_MODEL      = "facebook/musicgen-small"audio_bytes

# ── Placeholder label remapping ──────────────────────────────────────
# bhadresh model uses: sadness, joy, love, anger, fear, surprise
# We remap these to our 6 music moods
PLACEHOLDER_REMAP = {
    "joy":      "happy",
    "sadness":  "sad",
    "love":     "romantic",
    "anger":    "intense",
    "fear":     "intense",
    "surprise": "surprised",
}

# ── After fine-tuning, model outputs these directly ──────────────────
FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

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

if not USE_FINETUNED_MODEL:
    st.warning(
        "⚠️ Running with placeholder emotion model. "
        "Run the fine-tuning notebook and set `USE_FINETUNED_MODEL = True` for the final version.",
        icon="⚠️",
    )

st.markdown("---")

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model     = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    """
    ★ PIPELINE 2: text-classification
    Before fine-tuning : bhadresh-savani/distilbert-base-uncased-emotion (real HF model)
    After fine-tuning  : YOUR_HF_USERNAME/distilbert-go-emotions-music
    Fix : Previously used placeholder string that doesn't exist on HF Hub
          → now uses a real working model until fine-tuning is done
    """
    model_name = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
    )

@st.cache_resource(show_spinner=False)
def load_prompt_builder():
    """
    ★ PIPELINE 3: text2text-generation (pre-trained, no fine-tuning needed)
    Model : google/flan-t5-small — instruction-tuned, runs locally (~308MB)
    """
    return pipeline(
        "text2text-generation",
        model=FLANT5_MODEL,
        max_new_tokens=40,
    )

def get_hf_client():
    """HF Inference API client — used only for MusicGen audio generation."""
    token = st.secrets.get("HF_TOKEN", "")
    if not token:
        return None
    return InferenceClient(token=token)

def normalise_mood(label: str) -> str:
    """
    Normalise mood label to our 6 standard music moods.
    Handles both placeholder model output and fine-tuned model output.
    """
    label = label.lower()
    if USE_FINETUNED_MODEL:
        return label if label in FINETUNED_LABELS else "neutral"
    return PLACEHOLDER_REMAP.get(label, "neutral")

# ─────────────────────────────────────────────
#  PARALLEL INPUT SECTION
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📸 Upload your reel photo")
    uploaded = st.file_uploader(
        "Photo required to generate music",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with col_right:
    st.markdown("#### 💬 How are you feeling?")
    user_text = st.text_area(
        "Optional — describe your vibe",
        placeholder="e.g. missing summer so much...\nbest day ever with my girls\ncan't believe this view",
        height=160,
        label_visibility="collapsed",
    )
    st.caption("Leave empty → defaults to neutral mood")

st.markdown("---")

generate_btn = st.button(
    "🎵 Generate Background Music",
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

    # ── PIPELINE 1: BLIP (local, float16) ─────────────────────────────
    st.markdown("---")
    st.markdown('<p class="step-label">★ Pipeline 1 — image-to-text (BLIP, local)</p>',
                unsafe_allow_html=True)

    with st.spinner("Reading your photo scene..."):
        try:
            processor, blip_model = load_blip()  # ← unpack the tuple
            inputs = processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            caption = "a scenic photo"
            st.warning("Using fallback caption — app will continue.")

    st.markdown(f'<div class="caption-box">📝 Scene: {caption}</div>',
                unsafe_allow_html=True)

    # ── PIPELINE 2: Emotion Classifier (local) ─────────────────────────
    st.markdown(" ")
    model_label = "fine-tuned" if USE_FINETUNED_MODEL else "placeholder — replace after fine-tuning"
    st.markdown(f'<p class="step-label">★ Pipeline 2 — text-classification ({model_label})</p>',
                unsafe_allow_html=True)

    with st.spinner("Detecting mood from your text..."):
        try:
            if user_text.strip():
                classifier = load_emotion_classifier()
                raw_scores = classifier(user_text.strip())[0]
                raw_scores = sorted(raw_scores, key=lambda x: x["score"], reverse=True)

                # Normalise to our 6 music moods
                top_mood   = normalise_mood(raw_scores[0]["label"])
                top_score  = raw_scores[0]["score"]

                # Rebuild scores with normalised labels for display
                from collections import defaultdict
                merged = defaultdict(float)
                for s in raw_scores:
                    merged[normalise_mood(s["label"])] += s["score"]
                display_scores = sorted(merged.items(), key=lambda x: -x[1])
            else:
                top_mood      = "neutral"
                top_score     = 1.0
                display_scores = [("neutral", 1.0)]
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
            st.caption("*(no text input → defaulted to neutral)*")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    # ── PIPELINE 3: flan-t5-small prompt builder (local) ──────────────
    st.markdown(" ")
    st.markdown('<p class="step-label">★ Pipeline 3 — text2text-generation (flan-t5-small, local)</p>',
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

    # Replace get_hf_client() with this:
    def generate_music_api(prompt: str) -> bytes:
        token = st.secrets.get("HF_TOKEN", "")
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MUSICGEN_MODEL}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt},
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        return response.content

    with st.spinner("Composing your music clip (20–30 sec)..."):
        try:
            audio_bytes = generate_music_api(music_prompt)
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
    items = {
        "Scene caption":  caption,
        "Your text":      user_text.strip() if user_text.strip() else "*(none)*",
        "Detected mood":  f"{emoji} {top_mood.capitalize()} ({top_score*100:.1f}%)",
        "Music prompt":   music_prompt,
    }
    for k, v in items.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Pipeline Architecture")
    st.success("✅ **Pipeline 1** `image-to-text`\n\nBLIP-base (pre-trained)\n\nPhoto → caption\n\n*local, float16 ~250MB*")
    mode_note = "fine-tuned ⭐" if USE_FINETUNED_MODEL else "placeholder (pre fine-tuning)"
    st.success(f"✅ **Pipeline 2** `text-classification`\n\nDistilBERT ({mode_note})\n\nUser text → mood\n\n*local ~268MB*")
    st.success("✅ **Pipeline 3** `text2text-generation`\n\nflan-t5-small (pre-trained)\n\nCaption + mood → prompt\n\n*local ~308MB*")
    st.info("🎵 **MusicGen-small**\n\nHF Inference API\n\nPrompt → audio\n\n*not a pipeline*")
    st.markdown("---")
    total_ram = 250 + 268 + 308
    st.metric("Est. local RAM", f"~{total_ram}MB", delta="< 1GB limit ✅")
    st.markdown("---")
    st.markdown("### 🎭 Mood Classes")
    for mood, em in MOOD_EMOJI.items():
        st.markdown(f"{em} **{mood.capitalize()}**")
    st.markdown("---")
    st.caption("ISOM5240 Group Project · HuggingFace 🤗")
