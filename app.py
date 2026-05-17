
"""
ISOM5240 Group Project — VibeSound
Background Music Generator for Instagram Reels

Pipeline 1: image-to-text         (Salesforce/blip-image-captioning-base, HF Inference API)
Pipeline 2: text-classification   (DistilBERT fine-tuned on go_emotions remapped)
Pipeline 3: text2text-generation  (google/flan-t5-small, local)

Music Generation:
facebook/musicgen-small via Hugging Face Inference API
"""

import io
import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# Replace this with your actual Hugging Face model path after uploading your fine-tuned model.
# Example:
# FINETUNED_MODEL = "wenxueqing7/distilbert-go-emotions-music"
FINETUNED_MODEL = "MelodyWEN/distilbert-go-emotions-music"

BLIP_MODEL = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL = "google/flan-t5-small"
MUSICGEN_MODEL = "facebook/musicgen-small"

MOOD_FALLBACK = {
    "happy": "upbeat acoustic guitar, bright cheerful, fast tempo, major key",
    "sad": "slow melancholic piano, minor key, cinematic, emotional",
    "romantic": "soft romantic acoustic guitar, warm gentle, slow tempo",
    "intense": "dramatic orchestra, heavy drums, fast, dark, powerful",
    "surprised": "playful quirky ukulele, bouncy, dynamic, bright",
    "neutral": "calm ambient background, soft instrumental, peaceful",
}

MOOD_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "romantic": "❤️",
    "intense": "😠",
    "surprised": "😲",
    "neutral": "😐",
}

MOOD_COLOR = {
    "happy": "#FFD700",
    "sad": "#4A90D9",
    "romantic": "#E91E8C",
    "intense": "#E74C3C",
    "surprised": "#FF6B35",
    "neutral": "#888888",
}


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="VibeSound — Reel Music Generator",
    page_icon="🎵",
    layout="centered",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0D0D0D 0%, #1A0A2E 100%);
    }

    .title-text {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #E91E8C, #FF6B35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #aaa;
        font-size: 1rem;
        margin-top: -8px;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .step-label {
        color: #888;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .caption-box {
        color: #eee;
        font-style: italic;
        font-size: 1rem;
        padding: 12px;
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #E91E8C;
        border-radius: 4px;
    }

    .prompt-box {
        color: #eee;
        font-size: 0.95rem;
        padding: 12px;
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #FF6B35;
        border-radius: 4px;
    }

    .mood-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        color: #fff;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Generate background music for your Instagram Reel</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    """
    Pipeline 2:
    text-classification using fine-tuned DistilBERT.
    """
    return pipeline(
        "text-classification",
        model=FINETUNED_MODEL,
        return_all_scores=True,
    )


@st.cache_resource(show_spinner=False)
def load_prompt_builder():
    """
    Pipeline 3:
    text2text-generation using flan-t5-small.
    """
    return pipeline(
        "text2text-generation",
        model=FLANT5_MODEL,
        max_new_tokens=40,
    )


def get_hf_client():
    """
    Hugging Face Inference API client for BLIP and MusicGen.
    Make sure HF_TOKEN is added in Streamlit Cloud Secrets.
    """
    token = st.secrets.get("HF_TOKEN", "")

    if not token:
        return None

    return InferenceClient(token=token)


# ============================================================
# ROBUST OUTPUT PARSING
# ============================================================

def extract_caption(caption_result):
    """
    Robustly extract caption text from different Hugging Face InferenceClient return formats.

    Possible return formats include:
    - string
    - dict with generated_text
    - list of dict
    - object with generated_text attribute
    """

    if caption_result is None:
        return "a scenic photo"

    if isinstance(caption_result, str):
        return caption_result.strip()

    if isinstance(caption_result, dict):
        return str(caption_result.get("generated_text", "a scenic photo")).strip()

    if isinstance(caption_result, list) and len(caption_result) > 0:
        first_item = caption_result[0]

        if isinstance(first_item, dict):
            return str(first_item.get("generated_text", "a scenic photo")).strip()

        if hasattr(first_item, "generated_text"):
            return str(first_item.generated_text).strip()

        return str(first_item).strip()

    if hasattr(caption_result, "generated_text"):
        return str(caption_result.generated_text).strip()

    return str(caption_result).strip()


def extract_audio_bytes(audio_result):
    """
    Robustly extract audio bytes from Hugging Face InferenceClient text_to_audio result.
    """

    if audio_result is None:
        return None

    if isinstance(audio_result, bytes):
        return audio_result

    if isinstance(audio_result, bytearray):
        return bytes(audio_result)

    if hasattr(audio_result, "content"):
        return audio_result.content

    return audio_result


# ============================================================
# INPUT AREA
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📸 Upload your reel photo")
    uploaded = st.file_uploader(
        "Photo required",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with col_right:
    st.markdown("#### 💬 How are you feeling?")
    user_text = st.text_area(
        "Optional",
        placeholder=(
            "e.g. missing summer so much...\n"
            "best day ever with my girls\n"
            "can't believe this view 😍"
        ),
        height=160,
        label_visibility="collapsed",
    )
    st.caption("Optional — leave empty to default to neutral mood")

st.markdown("---")


generate_btn = st.button(
    "🎵 Generate My Background Music",
    type="primary",
    disabled=(uploaded is None),
    use_container_width=True,
)

if uploaded is None:
    st.caption("⬆️ Upload a photo to activate the generate button")


# ============================================================
# GENERATION PIPELINE
# ============================================================

if generate_btn and uploaded:

    client = get_hf_client()

    if client is None:
        st.error(
            "HF_TOKEN not found. Please add it under Streamlit Cloud → App Settings → Secrets."
        )
        st.stop()

    # ------------------------------------------------------------
    # Pipeline 1: BLIP Image Captioning
    # ------------------------------------------------------------

    st.markdown("---")
    st.markdown(
        '<p class="step-label">Pipeline 1 — Image-to-Text with BLIP</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Reading your photo scene..."):
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            caption_result = client.image_to_text(
                image=img_bytes.getvalue(),
                model=BLIP_MODEL,
            )

            caption = extract_caption(caption_result)

            if not caption:
                caption = "a scenic photo"

        except Exception as e:
            st.error(f"Image captioning failed: {type(e).__name__}: {e}")
            st.info(
                "Most likely causes: invalid HF_TOKEN, Hugging Face Inference API is busy, "
                "or the BLIP API returned a format that was not expected."
            )
            st.stop()

    st.markdown(
        f'<div class="caption-box">📝 Scene: {caption}</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Pipeline 2: Emotion Classification
    # ------------------------------------------------------------

    st.markdown(" ")
    st.markdown(
        '<p class="step-label">Pipeline 2 — Text Classification with Fine-tuned DistilBERT</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Detecting mood from your text..."):
        try:
            if user_text.strip():
                classifier = load_emotion_classifier()
                scores = classifier(user_text.strip())[0]
                scores = sorted(scores, key=lambda x: x["score"], reverse=True)

                top_mood = scores[0]["label"].lower()
                top_score = scores[0]["score"]

            else:
                top_mood = "neutral"
                top_score = 1.0
                scores = [{"label": "neutral", "score": 1.0}]

        except Exception as e:
            st.error(f"Mood detection failed: {type(e).__name__}: {e}")
            st.info(
                "Check whether FINETUNED_MODEL has been replaced with your actual Hugging Face model path."
            )
            st.stop()

    emoji = MOOD_EMOJI.get(top_mood, "🎶")
    color = MOOD_COLOR.get(top_mood, "#888888")

    col_mood, col_chart = st.columns([1, 2])

    with col_mood:
        st.markdown(
            f'<div class="mood-badge" style="background:{color};">'
            f'{emoji} {top_mood.capitalize()}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {top_score * 100:.1f}%")

        if not user_text.strip():
            st.caption("No text input → defaulted to neutral")

    with col_chart:
        if user_text.strip() and len(scores) >= 3:
            top3 = {
                s["label"].capitalize(): round(float(s["score"]), 3)
                for s in scores[:3]
            }
            st.bar_chart(top3, height=100)

    # ------------------------------------------------------------
    # Pipeline 3: Music Prompt Builder
    # ------------------------------------------------------------

    st.markdown(" ")
    st.markdown(
        '<p class="step-label">Pipeline 3 — Music Prompt Builder with FLAN-T5</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Building music prompt..."):
        try:
            prompt_builder = load_prompt_builder()

            instruction = (
                "Generate background music keywords for an Instagram reel. "
                f"Scene: {caption}. "
                f"Mood: {top_mood}. "
                "Output comma-separated music style keywords only, maximum 15 words."
            )

            raw_output = prompt_builder(instruction)[0]["generated_text"].strip()

            if raw_output and len(raw_output.split()) <= 20:
                music_prompt = raw_output
            else:
                music_prompt = MOOD_FALLBACK.get(
                    top_mood,
                    "calm ambient background music",
                )

        except Exception as e:
            st.warning(
                f"Prompt builder failed, using fallback prompt instead. Error: {type(e).__name__}: {e}"
            )
            music_prompt = MOOD_FALLBACK.get(
                top_mood,
                "calm ambient background music",
            )

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Music Generation: MusicGen
    # ------------------------------------------------------------

    st.markdown(" ")
    st.markdown(
        '<p class="step-label">Music Generation — MusicGen-small via HF Inference API</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Composing your music clip..."):
        try:
            audio_result = client.text_to_audio(
                text=music_prompt,
                model=MUSICGEN_MODEL,
            )

            audio_bytes = extract_audio_bytes(audio_result)

            if audio_bytes is None:
                raise ValueError("MusicGen returned empty audio output.")

        except Exception as e:
            st.error(f"Music generation failed: {type(e).__name__}: {e}")
            st.info(
                "This may happen if the Hugging Face Inference API is busy, "
                "the model is still loading, or the request times out."
            )
            st.stop()

    st.success("✅ Your background music is ready!")

    st.audio(audio_bytes, format="audio/wav")

    st.download_button(
        label="⬇️ Download Music (.wav)",
        data=audio_bytes,
        file_name=f"vibesound_{top_mood}.wav",
        mime="audio/wav",
    )

    # ------------------------------------------------------------
    # Result Summary
    # ------------------------------------------------------------

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Generation Summary")

    summary_data = {
        "Scene caption": caption,
        "Your text": user_text.strip() if user_text.strip() else "none",
        "Detected mood": f"{emoji} {top_mood.capitalize()} ({top_score * 100:.1f}%)",
        "Music prompt": music_prompt,
    }

    for key, value in summary_data.items():
        st.markdown(f"**{key}:** {value}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🔬 Pipeline Architecture")

    st.success(
        "✅ **Pipeline 1** `image-to-text`\n\n"
        "BLIP-base\n\n"
        "Photo → scene caption\n\n"
        "via HF Inference API"
    )

    st.success(
        "✅ **Pipeline 2** `text-classification`\n\n"
        "DistilBERT fine-tuned\n\n"
        "User text → mood label\n\n"
        "local model"
    )

    st.success(
        "✅ **Pipeline 3** `text2text-generation`\n\n"
        "flan-t5-small\n\n"
        "Caption + mood → music prompt\n\n"
        "local model"
    )

    st.info(
        "🎵 **MusicGen-small**\n\n"
        "Prompt → audio\n\n"
        "via HF Inference API"
    )

    st.markdown("---")
    st.markdown("### 🎭 Mood Classes")

    for mood, emoji in MOOD_EMOJI.items():
        st.markdown(f"{emoji} **{mood.capitalize()}**")

    st.markdown("---")
    st.caption("ISOM5240 Group Project · Hugging Face")
