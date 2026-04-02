"""
HangAI - modal_app.py
=====================
Run: modal deploy modal_app.py
All secrets are read from Modal secret store.

Changes in this version (v4):
  LLM — Gemma 3 4B replaced with Google Gemini 2.5 Flash API (free tier)
    • Eliminates GPU model loading (~30-60s saved per run)
    • Correction pass now uses chunked processing (15 lines/chunk) for reliability
    • Analysis pass uses Gemini's larger context window (28K chars vs 14K)
    • Removed transformers, bitsandbytes, accelerate dependencies
    • New env var required: GEMINI_API_KEY
  ASR — Provider quality upgrades
    • Deepgram: nova-2 → nova-3 (54% lower WER, native code-switching)
    • Deepgram: multi-language detection + keyword boosting for Hinglish terms
    • AssemblyAI: "best" speech model + auto code-switching + word boost
    • Segment grouping now splits on sentence-ending punctuation (।.?!)
    • Confidence-weighted dual-provider validation (low-conf → run 2nd provider)
  POST — Post-processing fixes
    • Unicode regex: added Arabic/Urdu + Gurmukhi script ranges
    • Hallucination filter: added cloud ASR patterns (filler words, lone periods)

Changes in this version (v3):
  ROUTER — ASR Router replaces direct Whisper call (asr_router.py)
    • Priority: Deepgram → AssemblyAI → ElevenLabs → local Whisper (fallback)
    • Per-provider API key rotation with cooldown TTL
    • Cloud providers skip chunking/overlap (they handle long audio natively)
    • Whisper model is only loaded if all cloud providers fail (saves GPU time)
    • All existing post-processing, diarization, LLM passes unchanged

Changes in this version (v2):
  A — Gemma 3 4B replaces Qwen2.5-7B (better Hinglish understanding)
  B — large-v3 replaces distil-large-v3 (higher accuracy, ~8-12% WER improvement)
  C — Hallucination filter catches CJK + more corruption patterns
  D — Transcript post-processing before LLM (merge, clean, dedupe)
  E — Few-shot examples in LLM prompt (anchors output quality)
  F — Auto language detection — language="hi" removed, Whisper now auto-detects
  G — VAD re-enabled with looser params (threshold 0.35, min_silence 500ms)
  H — Confidence threshold lowered 0.3 → 0.15 (preserves quiet/accented speech)
  I — LLM correction pass between ASR → analysis (fixes Hinglish + proper nouns)
  J — Chunking + 30s overlap for recordings > 5 minutes (prevents context loss)
"""

import modal
import os
import json
import re
from typing import Optional
from fastapi import Request

# ─────────────────────────────────────────────
# APP + IMAGE SETUP
# ─────────────────────────────────────────────

app = modal.App("hangai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "git"])
    .pip_install([
        "numpy<2.0",
        "faster-whisper==1.0.3",
        "pyannote.audio==3.3.2",
        "huggingface_hub<0.25",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "boto3==1.35.0",
        "supabase==2.7.4",
        "pydub==0.25.1",
        "fastapi==0.115.0",
        "python-multipart==0.0.12",
        # ── ROUTER: explicit requests pin for provider HTTP calls ──────────
        "requests>=2.31.0",
        # ── GEMINI: Google GenAI SDK for LLM correction + analysis ────────
        "google-genai>=1.0.0",
    ])
    # ── ROUTER: bundle asr_router.py into the container image ─────────────
    # asr_router.py must sit next to modal_app.py on your local machine.
    # Modal copies it into the container so "import asr_router" works inside
    # any function decorated with @app.function.
    .add_local_python_source("asr_router")
)

model_cache = modal.Volume.from_name("hangai-model-cache", create_if_missing=True)


from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ENDPOINT 1 — Upload audio chunk
# ─────────────────────────────────────────────

@web_app.post("/upload-chunk")
async def upload_chunk(request: Request):
    import boto3
    from botocore.config import Config

    session_id   = request.query_params.get("session_id")
    chunk_num    = int(request.query_params.get("chunk_num", 0))
    body         = await request.body()
    content_type = request.headers.get("content-type", "audio/wav")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-005",
    )

    ext = "raw"
    if content_type:
        parsed_ext = content_type.split("/")[-1].split(";")[0]
        if parsed_ext in ["wav", "webm", "mp4", "ogg", "m4a", "mpeg", "raw", "mkv"]:
            ext = parsed_ext
            
    key = f"sessions/{session_id}/chunk_{chunk_num:05d}.{ext}"

    s3.put_object(
        Bucket=os.environ["R2_BUCKET_NAME"],
        Key=key,
        Body=body,
        ContentType=content_type,
    )
    print(f"[HangAI] Stored: {key} ({len(body)} bytes)")
    return {"status": "ok", "key": key, "ext": ext}


# ─────────────────────────────────────────────
# ENDPOINT 2 — Trigger processing
# ─────────────────────────────────────────────

@web_app.post("/trigger-processing")
def trigger_processing(session_id: str, total_chunks: int, session_name: str = ""):
    process_recording.spawn(session_id, total_chunks, session_name)
    return {"status": "started", "session_id": session_id}

# ─────────────────────────────────────────────
# MOUNT ASGI APP
# ─────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("hangai-secrets")],
)
@modal.asgi_app(label="hangai-api")
def api():
    return web_app


def _preprocess_audio(input_path: str) -> str:
    """Normalize audio to 16kHz mono WAV with ffmpeg."""
    import subprocess
    output_path = input_path.rsplit('.', 1)[0] + '_preprocessed.wav'
    print(f"[HangAI] Preprocessing audio: {input_path} -> {output_path}")
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000',        # 16kHz sample rate (optimal for ASR)
        '-ac', '1',            # mono
        '-af', 'highpass=f=80,lowpass=f=8000,afftdn=nf=-25',  # denoise
        output_path,
    ], check=True, capture_output=True)
    return output_path



# ─────────────────────────────────────────────
# MAIN GPU FUNCTION
# ─────────────────────────────────────────────

WHISPER_CHUNK_DURATION = 300
WHISPER_CHUNK_OVERLAP  = 30


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/model-cache": model_cache},
    secrets=[modal.Secret.from_name("hangai-secrets")],
)
def process_recording(session_id: str, total_chunks: int, session_name: str = ""):
    import boto3
    import torch
    import io
    from botocore.config import Config
    from pydub import AudioSegment
    from pyannote.audio import Pipeline
    from google import genai
    from supabase import create_client

    # ── ROUTER imports (bundled via add_local_python_source) ─────────────
    from asr_router import ASRRouter, ASRAllProvidersFailed

    print(f"[HangAI] Processing session: {session_id} | chunks: {total_chunks}")

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    supabase.table("sessions").update({"status": "processing"}).eq("id", session_id).execute()

    # ── 1. Download from R2 ─────────────────────────────────────────────
    print("[HangAI] Downloading audio from R2...")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    # In v4.3, mobile optimization means the client uploads exactly ONE chunk directly.
    # We download this single chunk, whatever container it is (.webm, .mp4, etc).
    prefix = f"sessions/{session_id}/chunk_00000."
    response = s3.list_objects_v2(
        Bucket=os.environ["R2_BUCKET_NAME"],
        Prefix=prefix
    )

    if "Contents" not in response or len(response["Contents"]) == 0:
        supabase.table("sessions").update({"status": "error"}).eq("id", session_id).execute()
        raise ValueError("No audio chunk found in R2!")

    key = response["Contents"][0]["Key"]
    ext = key.split(".")[-1]
    audio_path = f"/tmp/{session_id}.{ext}"

    obj  = s3.get_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=key)
    data = obj["Body"].read()
    
    with open(audio_path, "wb") as f:
        f.write(data)
        
    # Get fast metadata directly without memory-heavy decoding 
    from pydub.utils import mediainfo
    info = mediainfo(audio_path)
    duration_seconds = float(info.get("duration", 0.0))
    file_size_mb = len(data) / 1_048_576

    print(f"[HangAI] File ready: {duration_seconds:.0f}s ({duration_seconds/60:.1f} min) | Native format: {ext} | Size: {file_size_mb:.2f} MB")

    # Preprocess audio to 16kHz WAV before routing to ASR
    try:
        processed_path = _preprocess_audio(audio_path)
        # Clean up the original downloaded file
        try:
            os.remove(audio_path)
        except OSError:
            pass
        audio_path = processed_path
    except Exception as e:
        print(f"[HangAI] Warning: Audio preprocessing failed ({e}), continuing with original file")

    # ═══════════════════════════════════════════════════════════════════════
    # ── 3. TRANSCRIPTION — ASR Router (cloud-first, Whisper fallback) ──
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Flow:
    #   A. Initialise router (reads key env vars, skips unconfigured providers)
    #   B. Try cloud providers in priority order: Deepgram → AssemblyAI → ElevenLabs
    #      • Cloud providers receive the full stitched WAV (they handle long audio)
    #      • No chunking/overlap needed on this side
    #      • On success: skip Whisper load entirely — saves GPU memory + time
    #   C. If ASRAllProvidersFailed → load Whisper + run existing chunked pipeline
    #   D. Either path produces `whisper_segs` (same format), then continues normally
    # ═══════════════════════════════════════════════════════════════════════

    print("[HangAI] Initialising ASR Router...")
    router = ASRRouter()

    whisper_segs:      Optional[list] = None
    detected_language: str            = "hi"
    language_prob:     float          = 0.0

    # ── 3A. Try cloud providers ──────────────────────────────────────────
    if router.providers:
        try:
            print(
                f"[HangAI] Attempting cloud ASR via "
                f"{[p.name for p in router.providers]}..."
            )
            asr_result = router.transcribe_with_validation(audio_path)

            # Normalise + clean segments (same pipeline as Whisper path)
            raw_segs       = asr_result["segments"]
            whisper_segs   = _postprocess_segments(raw_segs)
            detected_language = asr_result.get("detected_language", "hi")
            language_prob     = asr_result.get("language_probability", 0.0)

            print(
                f"[HangAI] ✓ Cloud ASR complete: {len(whisper_segs)} segments | "
                f"provider={asr_result['provider_used']} | "
                f"key={asr_result['key_used_suffix']} | "
                f"lang={detected_language} ({language_prob:.2f})"
            )

        except ASRAllProvidersFailed as e:
            print(
                f"[HangAI] ✗ All cloud ASR providers failed — "
                f"falling back to local Whisper.\n  Reason: {e}"
            )
            # whisper_segs remains None → triggers Whisper block below
    else:
        print("[HangAI] No cloud providers configured — using local Whisper directly")

    # ── 3B. Whisper fallback (only if cloud path didn't succeed) ─────────
    if whisper_segs is None:
        print("[HangAI] Loading Whisper large-v3 (fallback path)...")
        from faster_whisper import WhisperModel

        whisper = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
            download_root="/model-cache/whisper",
        )

        # ── Chunking for long recordings (unchanged from v2 / Change J) ──
        if duration_seconds > WHISPER_CHUNK_DURATION:
            print(
                f"[HangAI] Long recording ({duration_seconds/60:.1f} min) "
                f"— splitting into Whisper chunks..."
            )
            chunk_paths, chunk_offsets = _split_audio_for_whisper(
                audio_path,
                chunk_duration=WHISPER_CHUNK_DURATION,
                overlap=WHISPER_CHUNK_OVERLAP,
            )
            print(f"[HangAI] Created {len(chunk_paths)} Whisper chunks")
        else:
            chunk_paths   = [audio_path]
            chunk_offsets = [0.0]

        all_raw_segs       = []
        detected_language  = None
        language_prob      = 0.0
        prev_text_context  = ""

        for chunk_idx, (chunk_path, offset) in enumerate(zip(chunk_paths, chunk_offsets)):
            print(
                f"[HangAI] Whisper: transcribing chunk "
                f"{chunk_idx+1}/{len(chunk_paths)} (offset: {offset:.0f}s)..."
            )
            base_prompt = (
                "यह हिंदी और English मिली-जुली बातचीत है। "
                "लोग Hindi में बात करते हैं और बीच में English words भी use करते हैं। "
                "जैसे: Mobile, phone, okay, yes, no, WhatsApp, bhai, yaar. "
                "Names और places जैसे के हैं वैसे लिखो।"
            )
            initial_prompt = (
                f"{base_prompt}\n\nपिछली बातचीत: {prev_text_context}"
                if prev_text_context else base_prompt
            )

            segments, info = whisper.transcribe(
                chunk_path,
                task="transcribe",
                initial_prompt=initial_prompt,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "threshold":               0.35,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms":           200,
                },
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                hallucination_silence_threshold=2.0,
            )

            if detected_language is None:
                detected_language = info.language
                language_prob     = info.language_probability
            print(
                f"[HangAI] Whisper chunk {chunk_idx+1}: "
                f"lang={info.language} ({info.language_probability:.2f})"
            )

            chunk_segs = []
            for s in segments:
                text = s.text.strip()
                if _is_hallucination(text):
                    print(f"[HangAI] Filtered hallucination: '{text}'")
                    continue
                words = [
                    {
                        "word":  w.word,
                        "start": round(w.start + offset, 2),
                        "end":   round(w.end   + offset, 2),
                        "prob":  w.probability,
                    }
                    for w in (s.words or [])
                ]
                chunk_segs.append({
                    "start": round(s.start + offset, 2),
                    "end":   round(s.end   + offset, 2),
                    "text":  text,
                    "words": words,
                })

            all_raw_segs.extend(chunk_segs)

            if chunk_segs:
                tail_texts        = [s["text"] for s in chunk_segs[-3:]]
                prev_text_context = " ".join(tail_texts)

            if chunk_path != audio_path:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

        print(
            f"[HangAI] Whisper detected language: {detected_language} "
            f"(prob: {language_prob:.2f})"
        )
        print(f"[HangAI] Raw Whisper segments: {len(all_raw_segs)}")

        if len(chunk_paths) > 1:
            raw_segs = _dedupe_overlapping_segments(all_raw_segs)
            print(f"[HangAI] Segments after overlap dedup: {len(raw_segs)}")
        else:
            raw_segs = all_raw_segs

        whisper_segs = _postprocess_segments(raw_segs)
        print(f"[HangAI] Whisper segments after post-processing: {len(whisper_segs)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Everything below this line is UNCHANGED from v2.
    # whisper_segs is now populated from either cloud ASR or local Whisper.
    # ═══════════════════════════════════════════════════════════════════════

    # ── 5/6. Speaker diarization & Alignment ─────────────────────────────
    # Check if we already have native speaker labels from cloud ASR
    has_native_speakers = any("speaker" in w for seg in whisper_segs for w in seg.get("words", []))

    if has_native_speakers:
        print("[HangAI] Using native speaker labels from advanced ASR provider...")
        transcript_with_speakers = []
        for seg in whisper_segs:
            word_spks = [w["speaker"] for w in seg.get("words", []) if "speaker" in w]
            seg_spk = max(set(word_spks), key=word_spks.count) if word_spks else "0"
            transcript_with_speakers.append({**seg, "raw_speaker": seg_spk})

        # Map raw speakers (e.g. "0", "1", "A", "B") to "SPEAKER_00", "SPEAKER_01"
        unique_spks = []
        for s in transcript_with_speakers:
            if s["raw_speaker"] not in unique_spks:
                unique_spks.append(s["raw_speaker"])
        
        spk_map = {spk: f"SPEAKER_{i:02d}" for i, spk in enumerate(unique_spks)}

        for s in transcript_with_speakers:
            s["speaker"] = spk_map[s["raw_speaker"]]
            del s["raw_speaker"]
            
    else:
        print("[HangAI] Running local speaker diarization (Fallback)...")
        diarize = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ["HF_TOKEN"],
            cache_dir="/model-cache/pyannote",
        )
        diarize.to(torch.device("cuda"))
        diarization = diarize(audio_path)

        speaker_turns = [
            (turn.start, turn.end, spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
        ]

        # ── 6. Word-level speaker alignment (Fallback) ──────────────────────
        transcript_with_speakers = []
        for seg in whisper_segs:
            word_speakers = []
            for w in seg.get("words", []):
                best_spk, best_overlap = "SPEAKER_00", 0.0
                for t_start, t_end, spk in speaker_turns:
                    overlap = min(w["end"], t_end) - max(w["start"], t_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_spk     = spk
                word_speakers.append(best_spk)

            if word_speakers:
                speaker = max(set(word_speakers), key=word_speakers.count)
            else:
                mid = (seg["start"] + seg["end"]) / 2
                speaker = "SPEAKER_00"
                for t_start, t_end, spk in speaker_turns:
                    if t_start <= mid <= t_end:
                        speaker = spk
                        break

            transcript_with_speakers.append({**seg, "speaker": speaker})

    raw_full_transcript = "\n".join(
        f"[{s['speaker']} | {int(s['start']//60)}:{int(s['start']%60):02d}] {s['text']}"
        for s in transcript_with_speakers
    )

    speaker_time: dict = {}
    for seg in transcript_with_speakers:
        spk = seg["speaker"]
        speaker_time[spk] = speaker_time.get(spk, 0) + (seg["end"] - seg["start"])

    total_talk    = sum(speaker_time.values()) or 1
    speakers_data = [
        {
            "id":               spk,
            "label":            f"Speaker {i+1}",
            "percentage":       round((dur / total_talk) * 100),
            "duration_seconds": round(dur),
        }
        for i, (spk, dur) in enumerate(sorted(speaker_time.items(), key=lambda x: -x[1]))
    ]

    # ── 7. Initialise Gemini API client ──────────────────────────────────
    print("[HangAI] Initialising Gemini API client...")
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_MODELS = ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash-8b"]

    # ── 8. LLM correction pass (via Gemini API) ─────────────────────────
    print("[HangAI] Running LLM correction pass...")
    if len(raw_full_transcript.split()) > 50:
        full_transcript = _llm_correction_pass(gemini, GEMINI_MODELS, raw_full_transcript)
    else:
        full_transcript = raw_full_transcript
        print("[HangAI] Transcript too short for correction pass — skipping")

    # ── 9. LLM analysis pass (via Gemini API) ────────────────────────────
    MAX_CHARS = 28000   # Gemini handles much larger contexts than Gemma 4B
    if len(full_transcript) > MAX_CHARS:
        head = int(MAX_CHARS * 0.6)
        tail = MAX_CHARS - head
        transcript_for_llm = (
            full_transcript[:head]
            + "\n\n[... middle omitted ...]\n\n"
            + full_transcript[-tail:]
        )
    else:
        transcript_for_llm = full_transcript

    n_speakers = len(speakers_data)

    prompt = f"""You analyze real Indian conversation transcripts and return structured JSON.
The conversations are in Hindi, English, or Hinglish (code-switched). You understand all three.

IMPORTANT OUTPUT LANGUAGE: The "summary" field MUST be written in Hindi (Devanagari script).
Use natural Hindi with Hinglish terms where they appeared in the conversation.
All other text fields (key_moments descriptions, speaker_descriptions, mood) should also be in Hindi.
Keywords can stay in their original language. Action items can be in Hinglish.

Here are two examples of the quality expected:

EXAMPLE 1 — Input transcript snippet:
[SPEAKER_00 | 0:03] यार Zomato order किया था, 2 घंटे हो गए
[SPEAKER_01 | 0:08] bhai unka support toh bas "we are looking into it" bolte rehta hai
[SPEAKER_00 | 0:15] exactly! paise bhi gaye aur khana bhi nahi aaya

EXAMPLE 1 — Expected output:
{{
  "summary": "Two friends vented about a failed Zomato order that took over 2 hours and never arrived. SPEAKER_00 had placed the order and was frustrated after losing money with no food delivered. SPEAKER_01 shared the frustration, mocking Zomato's support team for their useless 'we are looking into it' response. The conversation was short but heated — a classic shared grievance over food delivery apps.",
  "key_moments": [{{"time_seconds": 15, "speaker": "SPEAKER_00", "description": "SPEAKER_00 summarised the double loss: 'paise bhi gaye aur khana bhi nahi aaya'"}}],
  "keywords": ["Zomato", "order", "2 ghante", "support", "paise"],
  "action_items": [],
  "speaker_descriptions": {{
    "SPEAKER_00": "The one who ordered — frustrated, direct, keeps bringing up the money lost",
    "SPEAKER_01": "Empathetic friend who validates and adds sarcasm about customer support"
  }},
  "mood": "Mutual frustration over a shared food delivery disaster",
  "language_mix": "60% Hindi, 40% English, heavy Hinglish throughout"
}}

EXAMPLE 2 — Input transcript snippet:
[SPEAKER_00 | 0:05] bhai kal interview tha Google mein
[SPEAKER_01 | 0:09] sach mein? kaisa gaya?
[SPEAKER_00 | 0:12] teen ghante ke baad bola rejected, coding round clear nahi hua

EXAMPLE 2 — Expected output:
{{
  "summary": "SPEAKER_00 shared that they had a Google interview the previous day that went badly — after a 3-hour process, they were rejected at the coding round. SPEAKER_01 was surprised and asked how it went, showing support. The conversation had a deflated energy, with SPEAKER_00 matter-of-factly delivering disappointing news without dramatising it.",
  "key_moments": [{{"time_seconds": 12, "speaker": "SPEAKER_00", "description": "Revealed the rejection — 3 hours of interview only to be told they didn't clear the coding round"}}],
  "keywords": ["Google", "interview", "coding round", "rejected", "teen ghante"],
  "action_items": [],
  "speaker_descriptions": {{
    "SPEAKER_00": "Calm and resigned about bad news — delivers a 3-hour rejection in one flat sentence",
    "SPEAKER_01": "Supportive, surprised, asks the right questions"
  }},
  "mood": "Quietly deflated — bad news delivered without drama",
  "language_mix": "70% Hindi, 30% English"
}}

---
NOW ANALYZE THIS REAL TRANSCRIPT ({n_speakers} speaker(s)):
---
{transcript_for_llm}
---

RULES:
- Be SPECIFIC — use actual words, phrases, names from the transcript above
- Quote Hindi/Hinglish phrases naturally in the summary where they add flavour
- NEVER write generic filler like "the speakers discussed various topics"
- key_moments: pick the most memorable/important actual moments with real timestamps
- action_items: ONLY if someone explicitly committed to doing something
- keywords: actual proper nouns and recurring phrases from THIS conversation only
- mood: one vivid descriptive phrase (see examples above for style)
- speaker_descriptions: how they actually sounded — energy, vocabulary, role

Respond ONLY with a valid JSON object. No markdown fences. No text before or after the JSON.

{{
  "summary": "...",
  "key_moments": [{{"time_seconds": 0, "speaker": "SPEAKER_00", "description": "..."}}],
  "keywords": ["..."],
  "action_items": ["..."],
  "speaker_descriptions": {{"SPEAKER_00": "..."}},
  "mood": "...",
  "language_mix": "..."
}}"""

    from google.genai import types as genai_types

    insights = {}
    for analysis_attempt in range(2):   # retry once on parse failure
        try:
            print(f"[HangAI] Calling Gemini API for analysis (attempt {analysis_attempt + 1})...")
            analysis_response = _call_gemini_with_fallback(
                gemini, 
                GEMINI_MODELS, 
                prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                )
            )
            response = analysis_response.text
            print(f"[HangAI] LLM analysis response (first 500 chars): {response[:500]}")
        except Exception as e:
            print(f"[HangAI] Gemini API analysis all models failed: {e}")
            response = ""

        insights = _extract_json(response)
        if insights:
            print(f"[HangAI] ✓ JSON parsed successfully — {len(insights)} keys")
            break
        else:
            print(f"[HangAI] JSON parse failed on attempt {analysis_attempt + 1}")

    if not insights:
        print("[HangAI] All JSON parse attempts failed — using raw response as summary")
        insights = {
            "summary":              response[:2000],
            "keywords":             [],
            "key_moments":          [],
            "action_items":         [],
            "mood":                 "",
            "language_mix":         "",
            "speaker_descriptions": {},
        }

    # ── 10. Save to Supabase ─────────────────────────────────────────────
    print("[HangAI] Saving to Supabase...")
    supabase.table("sessions").update({
        "status":               "completed",
        "duration_seconds":     int(duration_seconds),
        "transcript":           full_transcript,
        "summary":              insights.get("summary", ""),
        "key_moments":          insights.get("key_moments", []),
        "keywords":             insights.get("keywords", []),
        "action_items":         insights.get("action_items", []),
        "speaker_descriptions": insights.get("speaker_descriptions", {}),
        "speakers":             speakers_data,
        "word_count":           len(full_transcript.split()),
        "mood":                 insights.get("mood", ""),
        "language_mix":         insights.get("language_mix", ""),
    }).eq("id", session_id).execute()

    print(f"[HangAI] Done! Session {session_id} complete.")
    os.remove(audio_path)


# ─────────────────────────────────────────────
# LLM CORRECTION PASS (v4 — Gemini API, chunked)
# ─────────────────────────────────────────────

CORRECTION_CHUNK_SIZE = 15   # lines per chunk — keeps output reliable


def _call_gemini_with_fallback(client, model_ids, prompt, config=None):
    """
    Call Gemini API with model fallback if quota (429) is hit.
    """
    last_error = None
    for mid in model_ids:
        try:
            # If config is passed, use it; otherwise default call
            if config:
                return client.models.generate_content(model=mid, contents=prompt, config=config)
            else:
                return client.models.generate_content(model=mid, contents=prompt)
        except Exception as e:
            # Check for quota error (429) - genai SDK often wraps it or passes through
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                print(f"[HangAI] Quota hit for {mid}, falling back...")
                last_error = e
                continue
            raise e # Reraise other errors (auth, net, etc)
    
    raise last_error or RuntimeError("All Gemini models failed")


def _llm_correction_pass(gemini_client, model_ids: list, transcript: str) -> str:
    """
    Correct ASR errors in the transcript using Gemini API with model fallback.
    Processes in chunks of CORRECTION_CHUNK_SIZE lines for reliability.
    """
    original_lines = [l for l in transcript.split("\n") if l.strip()]

    if not original_lines:
        return transcript

    # Split into chunks
    chunks = []
    for i in range(0, len(original_lines), CORRECTION_CHUNK_SIZE):
        chunks.append(original_lines[i:i + CORRECTION_CHUNK_SIZE])

    corrected_all = []

    for chunk_idx, chunk_lines in enumerate(chunks):
        chunk_text = "\n".join(chunk_lines)

        correction_prompt = f"""You are a transcription editor. Fix ASR errors in this Hindi/English/Hinglish transcript chunk.

STRICT RULES:
1. Preserve EVERY line's format EXACTLY: [SPEAKER_XX | M:SS] original text
2. Fix ONLY clear ASR errors — mis-heard words, wrong transliterations, garbled proper nouns
3. Common Hinglish ASR errors to fix: "Kia"→"kya", "he "→"hai ", "kar raha he"→"kar raha hai", "Acha"→"Achha"
4. If an English word is clearly meant (context, surrounding words), keep it as English
5. Do NOT rephrase, summarize, or add any information not in the original
6. Do NOT merge or split lines — output EXACTLY {len(chunk_lines)} lines
7. Output ONLY the corrected lines — no preamble, no commentary, no markdown
8. Start immediately with the first [SPEAKER_ line

Transcript chunk to correct:
{chunk_text}"""

        try:
            response = _call_gemini_with_fallback(gemini_client, model_ids, correction_prompt)
            corrected = response.text.strip()
            corrected_lines = [l for l in corrected.split("\n") if l.strip()]

            # Validate chunk output
            if len(corrected_lines) < len(chunk_lines) * 0.8:
                print(
                    f"[HangAI] Correction chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"line count mismatch ({len(corrected_lines)} vs {len(chunk_lines)}) "
                    f"— using original chunk"
                )
                corrected_all.extend(chunk_lines)
                continue

            # Verify headers are preserved
            header_pattern = re.compile(r"^\[SPEAKER_\d+\s*\|\s*\d+:\d+\]")
            orig_headers = sum(1 for l in chunk_lines if header_pattern.match(l))
            corr_headers = sum(1 for l in corrected_lines if header_pattern.match(l))

            if orig_headers > 0 and corr_headers < orig_headers * 0.9:
                print(
                    f"[HangAI] Correction chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"headers stripped — using original chunk"
                )
                corrected_all.extend(chunk_lines)
                continue

            corrected_all.extend(corrected_lines)
            print(
                f"[HangAI] Correction chunk {chunk_idx + 1}/{len(chunks)}: "
                f"applied ({len(chunk_lines)} → {len(corrected_lines)} lines)"
            )

        except Exception as e:
            print(
                f"[HangAI] Correction chunk {chunk_idx + 1}/{len(chunks)} failed: {e} "
                f"— using original chunk"
            )
            corrected_all.extend(chunk_lines)

    print(
        f"[HangAI] Correction pass complete: "
        f"{len(original_lines)} → {len(corrected_all)} lines "
        f"({len(chunks)} chunks processed)"
    )
    return "\n".join(corrected_all)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response with multiple fallback strategies."""
    if not text:
        return {}

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    cleaned = cleaned.strip("`").strip()

    # Strategy 1: Direct parse
    try:
        import json
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find outermost { }
    try:
        depth = 0
        start_idx = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start_idx is not None:
                    try:
                        return json.loads(cleaned[start_idx : i + 1])
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass

    # Strategy 3: Fix common LLM JSON errors (like unescaped quotes in middle of strings)
    # This is a last resort and very basic.
    if "{" in cleaned and "}" in cleaned:
        try:
            # Try to just find the first { and last }
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last != -1:
                candidate = cleaned[first : last + 1]
                # Fix unescaped newlines
                candidate = candidate.replace("\n", "\\n")
                return json.loads(candidate)
        except Exception:
            pass

    return {}



# ─────────────────────────────────────────────
# AUDIO CHUNKING HELPERS (Change J — unchanged)
# ─────────────────────────────────────────────

def _split_audio_for_whisper(
    audio_path: str,
    chunk_duration: int = 300,
    overlap: int = 30,
) -> tuple[list[str], list[float]]:
    from pydub import AudioSegment

    audio      = AudioSegment.from_wav(audio_path)
    total_ms   = len(audio)
    chunk_ms   = chunk_duration * 1000
    overlap_ms = overlap * 1000
    step_ms    = chunk_ms - overlap_ms

    chunk_paths:   list[str]   = []
    chunk_offsets: list[float] = []

    i = 0
    chunk_idx = 0
    while i < total_ms:
        end_ms     = min(i + chunk_ms, total_ms)
        chunk      = audio[i:end_ms]
        chunk_path = audio_path.replace(".wav", f"_wchunk{chunk_idx:03d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
        chunk_offsets.append(i / 1000.0)
        print(f"[HangAI]   Chunk {chunk_idx}: {i/1000:.0f}s → {end_ms/1000:.0f}s → {chunk_path}")
        i += step_ms
        chunk_idx += 1
        if end_ms >= total_ms:
            break

    return chunk_paths, chunk_offsets


def _dedupe_overlapping_segments(segs: list) -> list:
    if len(segs) <= 1:
        return segs

    segs    = sorted(segs, key=lambda s: s["start"])
    deduped = [segs[0]]

    for seg in segs[1:]:
        prev         = deduped[-1]
        time_overlap = seg["start"] < prev["end"] + 3.0 and seg["start"] >= prev["start"]

        if time_overlap:
            prev_words = set(prev["text"].lower().split())
            seg_words  = set(seg["text"].lower().split())

            if prev_words and seg_words:
                union     = prev_words | seg_words
                intersect = prev_words & seg_words
                jaccard   = len(intersect) / len(union) if union else 0

                if jaccard > 0.45:
                    prev_conf = _avg_word_confidence(prev)
                    seg_conf  = _avg_word_confidence(seg)
                    if seg_conf > prev_conf:
                        deduped[-1] = seg
                    continue

        deduped.append(seg)

    return deduped


def _avg_word_confidence(seg: dict) -> float:
    words = seg.get("words", [])
    if not words:
        return 0.5
    return sum(w["prob"] for w in words) / len(words)


# ─────────────────────────────────────────────
# HALLUCINATION FILTER (Change C — unchanged)
# ─────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()

    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', stripped):
        return True
    if re.search(r'[\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf]', stripped):
        return True

    patterns = [
        r"^[\s\.\,\!\?\-\|]+$",
        r"♪|♫|🎵",
        r"\[.*?(music|applause|noise|silence|blank|inaudible|laughter).*?\]",
        r"^\s*(thank you|thanks|bye|goodbye|subscribe|like|share)\s*[\.\!]?\s*$",
        r"^\.{2,}$",
        r"^-{2,}$",
        r"^\(.*?\)$",
        # Cloud ASR-specific hallucinations:
        r"^\s*(um|uh|hmm|hm|ah|oh)\s*$",                    # filler words as standalone segments
        r"^\s*\.\s*$",                                        # just a period
    ]

    text_lower = stripped.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    words = text_lower.split()
    if len(words) >= 6:
        half   = len(words) // 2
        first  = " ".join(words[:half])
        second = " ".join(words[half:half * 2])
        if first == second:
            return True

    if len(words) >= 6:
        for n in range(2, 5):
            grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
            for i in range(len(grams)-1):
                if grams[i] == grams[i+1] == (grams[i+2] if i+2 < len(grams) else None):
                    return True

    return False


# ─────────────────────────────────────────────
# TRANSCRIPT POST-PROCESSING (Change D — unchanged)
# ─────────────────────────────────────────────

def _postprocess_segments(segs: list) -> list:
    if not segs:
        return segs

    def _get_speaker(s: dict) -> str:
        words = s.get("words", [])
        spks = [w["speaker"] for w in words if "speaker" in w]
        return max(set(spks), key=spks.count) if spks else None

    filtered = []
    for seg in segs:
        if seg.get("words"):
            avg_prob = sum(w["prob"] for w in seg["words"]) / len(seg["words"])
            if avg_prob < 0.15:
                print(f"[HangAI] Low confidence ({avg_prob:.2f}), dropping: '{seg['text']}'")
                continue
        filtered.append(seg)

    merged = []
    i = 0
    while i < len(filtered):
        seg        = filtered[i]
        word_count = len(seg["text"].split())
        
        # Only merge if the next segment exists AND belongs to the same speaker
        next_seg = filtered[i + 1] if i + 1 < len(filtered) else None
        same_speaker = next_seg and _get_speaker(seg) == _get_speaker(next_seg)
        
        if word_count <= 2 and same_speaker:
            merged_seg = {
                "start": seg["start"],
                "end":   next_seg["end"],
                "text":  seg["text"] + " " + next_seg["text"],
                "words": seg.get("words", []) + next_seg.get("words", []),
            }
            # Carry forward the speaker label explicitly if present
            merged_spk = _get_speaker(seg)
            if merged_spk:
                for w in merged_seg["words"]:
                    w["speaker"] = merged_spk
                    
            merged.append(merged_seg)
            i += 2
        else:
            merged.append(seg)
            i += 1

    merged = [s for s in merged if len(s["text"].strip()) > 1]

    deduped   = []
    prev_text = None
    for seg in merged:
        if seg["text"].strip() != prev_text:
            deduped.append(seg)
            prev_text = seg["text"].strip()

    cleaned = []
    for seg in deduped:
        clean_text = re.sub(
            r'[^\u0900-\u097f\u0964-\u0965\u0600-\u06ff\u0a00-\u0a7f\u0020-\u007e\u2000-\u206f\s]',
            '',
            seg["text"]
        ).strip()
        if clean_text:
            cleaned.append({**seg, "text": clean_text})

    return cleaned
