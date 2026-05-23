"""
ISOM5240 — VibeSound (1GB RAM + Streamlit Cloud)

Pipelines 1–3: local transformers (sequential load/unload).
Music: facebook/musicgen-small via public HF Space (free Inference API has no MusicGen provider).
"""
from __future__ import annotations

import gc
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

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

# ── CONFIG ───────────────────────────────────────────────────────────────────
USE_FINETUNED_MODEL = True
QUANTIZE_MOOD_MODEL = True

PLACEHOLDER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
FINETUNED_MODEL = "MelodyWEN7/vibesound-music-mood-classifier"

BLIP_MODEL = "Salesforce/blip-image-captioning-base"
FLANT5_MODEL = "google/flan-t5-small"
MUSICGEN_MODEL = "facebook/musicgen-small"
# Public HF Space (free; no paid Inference API). REST fn_index=0, no WebSockets.
MUSICGEN_SPACE_URL = "https://facebook-musicgen.hf.space"

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


def _read_audio_result(result) -> bytes:
    """Turn gradio Client output (path, bytes, tuple, dict) into wav bytes."""
    if isinstance(result, (list, tuple)):
        for item in result:
            try:
                return _read_audio_result(item)
            except (TypeError, ValueError):
                continue
        raise ValueError("No audio in gradio response tuple")

    if isinstance(result, dict):
        for key in ("path", "url", "name"):
            if key in result and result[key]:
                return _read_audio_result(result[key])
        raise ValueError(f"Unknown audio dict keys: {result.keys()}")

    if isinstance(result, bytes):
        return result

    if isinstance(result, str):
        if result.startswith("http"):
            r = requests.get(result, timeout=120)
            r.raise_for_status()
            return r.content
        path = Path(result)
        if path.is_file():
            return path.read_bytes()

    raise ValueError(f"Unsupported audio result type: {type(result)}")


def _space_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_sse_payload(line: str) -> dict | None:
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw:
        return None
    return json.loads(raw)


def _poll_gradio_call(event_url: str, headers: dict[str, str], timeout: float) -> list:
    """Poll Gradio queue over HTTP/SSE (no ws:// — avoids 'Unknown protocol: ws' on Cloud)."""
    deadline = time.monotonic() + timeout
    last_error = ""

    while time.monotonic() < deadline:
        with requests.get(event_url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                payload = _parse_sse_payload(raw_line)
                if not payload:
                    continue
                msg = payload.get("msg")
                if msg == "process_completed":
                    output = payload.get("output") or {}
                    data = output.get("data")
                    if data is None:
                        raise RuntimeError(f"Space returned no data: {output}")
                    return data
                if msg == "process_error":
                    raise RuntimeError(
                        payload.get("title")
                        or payload.get("message")
                        or str(payload)
                    )
                if msg in ("queue_full", "process_starts"):
                    continue
                if msg == "heartbeat":
                    continue
                last_error = str(payload)

        time.sleep(2)

    raise TimeoutError(
        f"MusicGen Space timed out after {int(timeout)}s"
        + (f" (last: {last_error})" if last_error else "")
    )


def _pick_audio_output(result) -> bytes:
    """MusicGen Spaces usually return (video, wav_path); prefer wav."""
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        for item in (result[1], result[0]):
            try:
                return _read_audio_result(item)
            except (TypeError, ValueError):
                continue
    return _read_audio_result(result)


def _start_gradio_job(
    call_url: str, headers: dict[str, str], prompt: str
) -> tuple[str, str]:
    """POST queue job; returns (poll_url, event_id)."""
    resp = requests.post(
        call_url,
        headers=headers,
        json={"data": [prompt, None]},
        timeout=120,
    )
    resp.raise_for_status()
    event_id = resp.json()
    if isinstance(event_id, dict):
        event_id = event_id.get("event_id")
    if not event_id:
        raise RuntimeError(f"Unexpected Space response: {resp.text[:200]}")
    return f"{call_url}/{event_id}", str(event_id)


def generate_music_hf_space(prompt: str) -> bytes:
    """
    MusicGen is NOT on Hugging Face free serverless Inference API
    (availableInferenceProviders is empty — GPU-heavy text-to-audio).

    Uses the public facebook/MusicGen Space via HTTP queue API only (no gradio_client/ws).
    Endpoint 0: text + optional melody -> (video, wav).
    """
    token = get_hf_token() or None
    headers = _space_headers(token or None)
    base = MUSICGEN_SPACE_URL
    # Gradio 4 queue API (fn_index 0); fallback for older Space builds
    call_candidates = (
        f"{base}/gradio_api/call/0",
        f"{base}/call/0",
        f"{base}/run/0",
    )

    errors: list[str] = []
    for call_url in call_candidates:
        try:
            poll_url, _ = _start_gradio_job(call_url, headers, prompt)
            data = _poll_gradio_call(poll_url, headers, timeout=600)
            return _pick_audio_output(data)
        except Exception as e:
            errors.append(f"{call_url}: {e}")

    raise RuntimeError(
        "MusicGen Space unreachable. "
        + " | ".join(errors[-3:])
        + " — wait 1–2 min if the Space is cold and retry."
    )


# ── PAGE ─────────────────────────────────────────────────────────────────────
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
    /* Avoid double-label feel on file uploader */
    [data-testid="stForm"] [data-testid="stFileUploader"] label p {
        font-size: 1rem; font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Background music for your Instagram Reel</p>',
    unsafe_allow_html=True,
)

# ── SINGLE INPUT FORM (one photo box, one text box) ─────────────────────────
with st.form("vibesound_input", border=True):
    st.markdown("**Upload once, then generate**")
    col_photo, col_text = st.columns(2)
    with col_photo:
        uploaded = st.file_uploader(
            "Reel photo",
            type=["jpg", "jpeg", "png"],
            key="reel_photo",
        )
    with col_text:
        user_text = st.text_area(
            "How are you feeling? (optional)",
            placeholder="e.g. best day ever with my girls",
            height=120,
            key="vibe_text",
        )
    submitted = st.form_submit_button(
        "🎵 Generate Background Music",
        type="primary",
        use_container_width=True,
    )

if uploaded is not None:
    preview = Image.open(uploaded).convert("RGB")
    st.image(preview, caption="Preview", width=280)

if submitted:
    if uploaded is None:
        st.error("Please upload a reel photo first.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")

    st.markdown("---")
    st.markdown(
        '<p class="step-label">★ Pipeline 1 — image-to-text (BLIP)</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reading your photo..."):
        try:
            caption = run_blip(image)
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            caption = "a scenic photo"
            st.warning("Using fallback caption.")

    st.markdown(
        f'<motion class="caption-box">📝 Scene: {caption}</div>'.replace(
            '<motion class="caption-box">', '<motion class="caption-box">'
        ),
        unsafe_allow_html=True,
    )

    mood_label = "fine-tuned RoBERTa" if USE_FINETUNED_MODEL else "placeholder"
    st.markdown(
        f'<p class="step-label">★ Pipeline 2 — text-classification ({mood_label})</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Detecting mood..."):
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
            f'<motion class="mood-badge" style="background:{color};">'
            f'{emoji} {top_mood.capitalize()}</motion>'.replace("<motion", "<div").replace(
                "</motion>", "</motion>".replace("motion", "motion")
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {top_score * 100:.1f}%")
        if not user_text.strip():
            st.caption("*(no text → neutral)*")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    st.markdown(
        '<p class="step-label">★ Pipeline 3 — text2text-generation (flan-t5-small)</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Building music prompt..."):
        try:
            music_prompt = run_prompt_builder(caption, top_mood)
        except Exception:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")

    st.markdown(
        f'<motion class="prompt-box">🎼 Music prompt: {music_prompt}</motion>'.replace(
            '<motion class="prompt-box">', '<motion class="prompt-box">'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="step-label">Music — MusicGen-small (HF Space API)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Model `{MUSICGEN_MODEL}` · free HF Inference API has no audio provider · using public Space"
    )
    with st.spinner("Composing music via HF Space (may take 1–2 min if sleeping)..."):
        try:
            audio_bytes = generate_music_hf_space(music_prompt)
        except Exception as e:
            st.error(f"Music generation failed: {e}")
            st.info(
                "**No paid API key required** for the public MusicGen Space. "
                "Add a **free** `HF_TOKEN` in Streamlit secrets (Hub → Settings → Access Tokens) "
                "for queue priority + your gated mood model. "
                "Paid [Inference Endpoints](https://huggingface.co/docs/inference-endpoints) "
                "only if you need a dedicated GPU API."
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

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Summary")
    for k, v in {
        "Scene": caption,
        "Text": user_text.strip() or "*(none)*",
        "Mood": f"{emoji} {top_mood.capitalize()} ({top_score * 100:.1f}%)",
        "Prompt": music_prompt,
    }.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown("</motion>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Architecture")
    st.success("**P1** BLIP image→text (local)")
    st.success("**P2** RoBERTa mood (local)")
    st.success("**P3** flan-t5 prompt (local)")
    st.warning(
        "**MusicGen** via HF Space\n\n"
        "Not on free Inference API\n\n"
        "`endpoints_compatible` only"
    )
    st.metric("Peak RAM", "~600MB", delta="sequential")
    st.markdown("---")
    for mood, em in MOOD_EMOJI.items():
        st.markdown(f"{em} {mood.capitalize()}")
    st.caption("ISOM5240 · HuggingFace 🤗")
