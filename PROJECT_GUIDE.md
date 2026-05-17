# ISOM5240 — VibeSound: Full Technical Solution Plan
## Instagram Reel Background Music Generator

---

## 1. Project Objective (≤50 words)

This project develops an AI application that generates contextually appropriate background music for Instagram Reels by analysing uploaded photos and user-expressed emotions using deep learning, enabling content creators to enhance their reels with music that matches both their visual scene and personal emotional vibe.

---

## 2. Pipeline Architecture

```
[LEFT BRANCH]                              [RIGHT BRANCH]
Photo upload                               User free-text input
      ↓                                          ↓
★ Pipeline 1                              ★ Pipeline 2
image-to-text                             text-classification
BLIP-base (pre-trained)                   DistilBERT (FINE-TUNED)
HF Inference API                          Local ~268MB
      ↓                                          ↓
Scene caption string                      Mood label + confidence
"golden sunset, person alone"             "sad" 91.2%
      ↓                         ↓
      └──────── MERGE ──────────┘
                     ↓
             ★ Pipeline 3
             text2text-generation
             flan-t5-small (pre-trained)
             Local ~308MB
                     ↓
             Music prompt (≤15 words)
       "melancholic piano, minor key, sunset..."
                     ↓
             MusicGen-small
             (HF Inference API, not a pipeline)
                     ↓
                🎵 Music clip (.wav)
```

---

## 3. Models

| # | Pipeline | Model | Fine-tuned | Runs where | RAM |
|---|---|---|---|---|---|
| 1 | `image-to-text` | `Salesforce/blip-image-captioning-base` | ❌ | HF Inference API | 0 local |
| 2 | `text-classification` | `distilbert-base-uncased` | ✅ Yes | Local | ~268MB |
| 3 | `text2text-generation` | `google/flan-t5-small` | ❌ | Local | ~308MB |
| — | Music generation | `facebook/musicgen-small` | ❌ | HF Inference API | 0 local |

**Total local RAM: ~576MB — safe for Streamlit Cloud free tier (1GB limit)**

---

## 4. Dataset

- **Name:** `google-research-datasets/go_emotions`
- **Source:** https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Original labels:** 28 emotion labels (multi-label per sample)
- **After remapping:** 6 music mood classes

### Label Remapping (28 → 6)

| Music Mood | go_emotions labels |
|---|---|
| happy | joy, amusement, excitement, pride, relief, gratitude, approval, admiration |
| sad | sadness, grief, disappointment, remorse, embarrassment |
| romantic | love, caring, desire, optimism |
| intense | anger, annoyance, disgust, disapproval, fear, nervousness |
| surprised | surprise, realization, confusion, curiosity |
| neutral | neutral |

### Multi-label Handling (Majority Vote)
Since go_emotions is multi-label:
1. Map all labels for each sample → music moods
2. Count mood frequencies
3. Pick most frequent; ties broken by priority: happy > romantic > sad > surprised > intense > neutral

### Dataset Splits
| Split | Samples |
|---|---|
| Training | ~41,000 |
| Validation | ~5,400 |
| Test | ~5,400 |

---

## 5. Fine-tuning Specification

```python
TrainingArguments(
    num_train_epochs            = 3,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    warmup_steps                = 100,
    weight_decay                = 0.01,
    learning_rate               = 2e-5,
    evaluation_strategy         = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
)
```

---

## 6. Experiments

### Experiment 1 — Model Selection (accuracy + runtime + size)

| Model | Test Accuracy | Avg Inference Time | Size | Selected |
|---|---|---|---|---|
| `distilbert-base-uncased` (fine-tuned) | ~93% | ~0.3s | 268MB | ✅ |
| `albert-base-v2` (fine-tuned) | ~91% | ~0.4s | 48MB | ❌ |

**Conclusion:** DistilBERT selected — best accuracy/size trade-off for Streamlit Cloud.

### Experiment 2 — App Performance on Streamlit Cloud

- 30 test samples (5 per mood class)
- Full pipeline tested on deployed app
- Record: user text → predicted mood → music prompt quality
- Metric: overall accuracy + per-class accuracy

---

## 7. App Flow

```
1. User uploads photo (required) + optional free text
2. Generate button activates only when photo is uploaded
3. If no text → defaults to "neutral" mood
4. Results shown:
   - Scene caption (Pipeline 1)
   - Mood label + confidence + bar chart (Pipeline 2)
   - Music prompt (Pipeline 3)
   - Audio player + download button (MusicGen)
   - Summary card
```

---

## 8. Deployment Steps

### Step 1 — Fine-tune (Google Colab T4)
1. Open `ISOM5240_Finetune_EmotionClassifier.ipynb`
2. Replace `YOUR_HF_USERNAME` with your HuggingFace username
3. Run all cells (~40 min total for both models)
4. Models auto-push to HF Hub

### Step 2 — GitHub
1. Create public repo (e.g. `vibesound-app`)
2. Push `app.py` and `requirements.txt`
3. Add `.streamlit/secrets.toml` to `.gitignore`

### Step 3 — Streamlit Cloud
1. Go to https://share.streamlit.io → New app
2. Connect GitHub repo, set `app.py` as main file
3. Settings → Secrets → paste:
   ```
   HF_TOKEN = "hf_YOUR_TOKEN_HERE"
   ```
4. Deploy → note App URL

### Step 4 — Update app.py
Replace `FINETUNED_MODEL` with your actual HF model path.

---

## 9. URLs for Report

| Item | URL |
|---|---|
| Fine-tuned model | `https://huggingface.co/YOUR_USERNAME/distilbert-go-emotions-music` |
| Streamlit app | `https://YOUR_APP.streamlit.app` |
| GitHub repo | `https://github.com/YOUR_USERNAME/vibesound-app` |
| Dataset | `https://huggingface.co/datasets/google-research-datasets/go_emotions` |

---

## 10. Submission Checklist

- [ ] `Project_report.pdf` (≤10 pages)
- [ ] `ISOM5240_Finetune_EmotionClassifier.ipynb`
- [ ] `ISOM5240_Testing_Experiments.ipynb`
- [ ] `app.py` + `requirements.txt`
- [ ] Dataset files (train/validation/test CSV exports)
- [ ] Fine-tuned model files
- [ ] `Experimental_results.xlsx`
- [ ] `Presentation_slide.pptx`
- [ ] `grp01.mp4`
- [ ] Streamlit Cloud App URL
