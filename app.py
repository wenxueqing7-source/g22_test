"""
ISOM5240 Group Project — VibeSound (1GB RAM edition)
Background Music Generator for Instagram Reels

Sequential model loading — only one local model in RAM at a time.

HuggingFace Pipelines (transformers.pipeline):
  ★ Pipeline 1: image-to-text         (Salesforce/blip-image-captioning-base, LOCAL float16)
  ★ Pipeline 2: text-classification   (MelodyWEN7/vibesound-music-mood-classifier, RoBERTa)
  ★ Pipeline 3: text2text-generation  (google/flan-t5-small, pre-trained, float16)

Music Generation (HF Inference API — not a pipeline):
  facebook/musicgen-small
"""
from __future__ import annotations

import gc
import os
from collections import defaultdict

import requests
import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
USE_FINETUNED_MODEL = True
QUANTIZE_MOOD_MODEL = True  # dynamic int8 — ~500MB RoBERTa → ~200–250MB

PLACEHOLDER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
FINETUNED_MODEL = "MelodyWEN7/vibesound-music-mood-classifier"

BLIP_MODEL = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL = "google/flan-t5-small"
MUSICGEN_MODEL = "facebook/musicgen-small"

PLACEHOLDER_REMAP = {
    "joy": "happy",
    "sadness": "sad",
    "love": "romantic",
    "anger": "intense",
    "fear": "intense",
    "surprise": "surprised",
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


def get_hf_token() -> str:
    try:
        token = st.secrets.get("HF_TOKEN", "")
        if token:
            return token
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("HF_TOKEN", "")


def _free(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()


def normalise_mood(label: str) -> str:
    label = label.lower()
    if USE_FINETUNED_MODEL:
        return label if label in FINETUNED_LABELS else "neutral"
    return PLACEHOLDER_REMAP.get(label, "neutral")


@torch.inference_mode()
def run_blip(image: Image.Image) -> str:
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    _free(model, processor, inputs, out)
    return caption


@torch.inference_mode()
def run_mood(text: str) -> tuple[str, float, list[tuple[str, float]]]:
    if not text.strip():
        return "neutral", 1.0, [("neutral", 1.0)]

    model_name = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if QUANTIZE_MOOD_MODEL and USE_FINETUNED_MODEL:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=-1,
    )
    raw_scores = sorted(clf(text.strip())[0], key=lambda x: x["score"], reverse=True)
    _free(clf, model, tokenizer)

    top_mood = normalise_mood(raw_scores[0]["label"])
    top_score = raw_scores[0]["score"]

    merged: defaultdict[str, float] = defaultdict(float)
    for s in raw_scores:
        merged[normalise_mood(s["label"])] += s["score"]
    display_scores = sorted(merged.items(), key=lambda x: -x[1])
    return top_mood, top_score, display_scores


@torch.inference_mode()
def run_prompt_builder(caption: str, top_mood: str) -> str:
    gen = pipeline(
        "text2text-generation",
        model=FLANT5_MODEL,
        torch_dtype=torch.float16,
        max_new_tokens=40,
        device=-1,
    )
    instruction = (
        f"Generate background music keywords for an Instagram reel. "
        f"Scene: {caption}. "
        f"Mood: {top_mood}. "
        f"Output comma-separated music style keywords only, maximum 15 words."
    )
    raw_output = gen(instruction)[0]["generated_text"].strip()
    _free(gen)
    if raw_output and len(raw_output.split()) <= 20:
        return raw_output
    return MOOD_FALLBACK.get(top_mood, "calm ambient music")


def generate_music_api(prompt: str) -> bytes:
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing — add to Streamlit secrets or export HF_TOKEN locally."
        )
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
#  UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator",
    page_icon="🎵",
    layout="centered",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Generate background music for your Instagram Reel</p>',
    unsafe_allow_html=True,
)
st.caption("1GB mode: one model loaded at a time, then released from RAM.")

st.markdown("---")

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
    st.caption("Leave empty → neutral mood (skips RoBERTa — saves ~500MB)")

st.markdown("---")

generate_btn = st.button(
    "🎵 Generate Background Music",
    type="primary",
    disabled=(uploaded is None),
    use_container_width=True,
)

if uploaded is None:
    st.caption("⬆️ Upload a photo to activate the generate button")

if generate_btn and uploaded:
    st.markdown("---")
    st.markdown(
        '<p class="step-label">★ Pipeline 1 — image-to-text (BLIP, local float16)</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reading your photo scene..."):
        try:
            caption = run_blip(image)
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            caption = "a scenic photo"
            st.warning("Using fallback caption.")

    st.markdown(
        f'<div class="caption-box">📝 Scene: {caption}</div>',
        unsafe_allow_html=True,
    )

    mood_label = (
        "fine-tuned RoBERTa"
        if USE_FINETUNED_MODEL
        else "placeholder DistilBERT"
    )
    st.markdown(
        f'<p class="step-label">★ Pipeline 2 — text-classification ({mood_label})</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Detecting mood from your text..."):
        try:
            top_mood, top_score, display_scores = run_mood(user_text)
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
        st.caption(f"Confidence: {top_score * 100:.1f}%")
        if not user_text.strip():
            st.caption("*(no text input → defaulted to neutral)*")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    st.markdown(
        '<p class="step-label">★ Pipeline 3 — text2text-generation (flan-t5-small, local float16)</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Building music prompt..."):
        try:
            music_prompt = run_prompt_builder(caption, top_mood)
        except Exception:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="step-label">Music Generation — MusicGen-small (HF Inference API)</p>',
        unsafe_allow_html=True,
    )
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

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Generation Summary")
    items = {
        "Scene caption": caption,
        "Your text": user_text.strip() if user_text.strip() else "*(none)*",
        "Detected mood": f"{emoji} {top_mood.capitalize()} ({top_score * 100:.1f}%)",
        "Music prompt": music_prompt,
    }
    for k, v in items.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Pipeline Architecture")
    st.success(
        "✅ **Pipeline 1** `image-to-text`\n\n"
        "BLIP-base (pre-trained)\n\nPhoto → caption\n\n*local float16, sequential*"
    )
    mode_note = "fine-tuned RoBERTa ⭐" if USE_FINETUNED_MODEL else "placeholder"
    quant_note = " + int8 quant" if QUANTIZE_MOOD_MODEL and USE_FINETUNED_MODEL else ""
    st.success(
        f"✅ **Pipeline 2** `text-classification`\n\n"
        f"{mode_note}{quant_note}\n\nUser text → mood\n\n*sequential, ~250–500MB peak*"
    )
    st.success(
        "✅ **Pipeline 3** `text2text-generation`\n\n"
        "flan-t5-small (pre-trained)\n\nCaption + mood → prompt\n\n*local float16, sequential*"
    )
    st.info(
        "🎵 **MusicGen-small**\n\nHF Inference API\n\nPrompt → audio\n\n*not a pipeline*"
    )
    st.markdown("---")
    st.metric("Peak local RAM", "~550–700MB", delta="sequential < 1GB ✅")
    st.markdown("---")
    st.markdown("### 🎭 Mood Classes")
    for mood, em in MOOD_EMOJI.items():
        st.markdown(f"{em} **{mood.capitalize()}**")
    st.markdown("---")
    st.caption("ISOM5240 Group Project · HuggingFace 🤗")
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
