# HangAI — Hinglish Conversation Intelligence System

> Record your hangouts. Get back decisions, action items, and who said what — automatically.

HangAI is an end-to-end audio intelligence pipeline that turns hours of casual Hinglish (Hindi + English) conversation into structured summaries with speaker attribution, timestamps, and sentiment — at near-zero cost.

---

## The Problem

You spend 4 hours at a hangout. Important things get discussed — plans, decisions, ideas. Two days later, nobody remembers exactly what was said or who committed to what. Notes don't get taken. Voice memos don't get replayed.

HangAI solves this by recording passively from your phone, transcribing with multi-provider fallback, diarizing speakers, and producing an LLM-generated summary you can actually act on.

---

## Architecture

```
Phone (Chrome) → Cloudflare R2 (audio chunks)
                      ↓
              Modal GPU Backend
                ├── ASR Pipeline (AssemblyAI → Deepgram → ElevenLabs fallback)
                ├── Speaker Diarization (pyannote 3.1)
                └── LLM Summarization (Qwen2.5-7B)
                      ↓
              Supabase (structured output)
                      ↓
              GitHub Pages (dashboard)
```

---

## Key Features

- **Multi-provider ASR with dynamic routing** — Benchmarked AssemblyAI, Deepgram, and ElevenLabs on code-switched Hinglish audio; routes to best-performing provider automatically, with fallback if rate limits hit. ~18% accuracy improvement over single-provider.
- **Speaker diarization** — Up to 4 speakers identified and attributed across the full transcript using pyannote 3.1.
- **LLM summarization** — 4–5 hours of audio compressed to ~30 minute read: key decisions, action items, speaker-level sentiment.
- **Zero-cost architecture** — Runs entirely on free tiers (Supabase, Cloudflare R2, GitHub Pages, Modal). GPU cost ~$1–4/month for weekly use.
- **Passive recording** — Record from your phone browser, pocket it, forget about it.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | HTML, Vanilla JS, GitHub Pages |
| Audio Storage | Cloudflare R2 |
| ASR | AssemblyAI, Deepgram, ElevenLabs |
| Diarization | pyannote/speaker-diarization-3.1 |
| GPU Backend | Modal (A10G) |
| LLM | Qwen2.5-7B-Instruct |
| Database | Supabase (PostgreSQL) |

---

## Setup

Full step-by-step setup is in [SETUP.md](./SETUP.md).

**Quick summary:**
1. Create Supabase project → run schema SQL
2. Create Cloudflare R2 bucket → configure CORS
3. Get Hugging Face token → accept pyannote model terms
4. Deploy backend to Modal with secrets
5. Configure `index.html` → deploy to GitHub Pages

Estimated setup time: 45–60 minutes.

---

## Usage

**Before the hangout:**
```
modal deploy modal_app.py
```

**At the hangout:**  
Open your GitHub Pages URL on Chrome → tap Start Recording → pocket your phone.

**After:**  
Open dashboard → wait 20–30 min → read the summary.

---

## Cost Breakdown (1 session/week)

| Service | Cost |
|---|---|
| Supabase | $0 |
| Cloudflare R2 | $0 (under 10GB free tier) |
| GitHub Pages | $0 |
| Modal GPU | ~$1–4/month |
| **Total** | **~$1–4/month** |

---

## Limitations

- Recording requires Chrome browser with mic permissions
- Chrome tab must stay open during recording (background tab is fine)
- Diarization accuracy drops with heavy background noise or 4+ overlapping speakers
- Hinglish transcription quality depends on audio clarity and which provider handles the chunk

---

## Author

**Parshva Karani** — AI Engineer  
[LinkedIn](https://linkedin.com/in/parshva-karani) · [GitHub](https://github.com/parshvak26)
