# HangAI — Complete Setup Guide
Everything you need to do, once, in order.

---

## What you need before starting
- A laptop (Windows/Mac/Linux — any works)
- Python 3.11 installed → https://python.org/downloads
- Git installed → https://git-scm.com
- A phone with Chrome browser
- Credit card (for Cloudflare R2 — won't be charged, just required)

---

## STEP 1 — Create a Supabase project (your database)

1. Go to https://supabase.com → Sign up (free)
2. Click "New Project" → give it any name → set a DB password (save it) → pick any region
3. Wait 2 minutes for it to create
4. Click "SQL Editor" in left sidebar → "New query"
5. Paste everything from `supabase_schema.sql` → click "Run"
6. Go to "Settings" → "API" → copy these two values:
   - **Project URL** → looks like `https://abcdefgh.supabase.co`
   - **anon public key** → a long string starting with `eyJ...`
7. Save both. You'll paste them into `index.html` later.
8. Also copy the **service_role key** (it's below the anon key). Save this too — it goes into Modal secrets.

---

## STEP 2 — Create Cloudflare R2 bucket (your audio storage)

1. Go to https://cloudflare.com → Sign up (free)
2. In dashboard left sidebar, click "R2 Object Storage"
3. Click "Create bucket" → name it `hangai-audio` → click Create
4. In the bucket, click "Settings" → scroll to "CORS Policy" → click "Edit CORS policy"
5. Paste this CORS config and save:

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

6. Go back to R2 main page → click "Manage R2 API Tokens"
7. Click "Create API Token" → give it any name → set permissions to "Edit" → click Create
8. Copy and save these three values:
   - **Access Key ID** → looks like `abc123...`
   - **Secret Access Key** → a long random string (only shown once!)
   - **Endpoint URL** → looks like `https://ACCOUNT_ID.r2.cloudflarestorage.com`

---

## STEP 3 — Get a Hugging Face token (for the diarization model)

The pyannote speaker detection model needs HuggingFace access.

1. Go to https://huggingface.co → Sign up (free)
2. Go to your profile → Settings → Access Tokens → "New token"
3. Name it "hangai" → select "Read" → Create
4. Copy the token (starts with `hf_...`)
5. Now accept the model terms:
   - Go to https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"
   - Also go to https://huggingface.co/pyannote/segmentation-3.0
   - Click "Agree and access repository"

Without accepting terms, the model download will fail with a 403 error.

---

## STEP 4 — Set up Modal (your GPU + backend)

1. Go to https://modal.com → Sign up (free)
2. Install Modal on your laptop. Open terminal and run:
   ```
   pip install modal
   ```
3. Log in to Modal:
   ```
   modal token new
   ```
   This opens a browser to authenticate. Click "Create token". Done.

4. Create a Modal secret with all your API keys. Run this command
   (replace each value with your actual keys from steps 1-3):
   ```
   modal secret create hangai-secrets \
     SUPABASE_URL="https://XXXXXXXXXX.supabase.co" \
     SUPABASE_SERVICE_KEY="eyJ..." \
     R2_ENDPOINT_URL="https://ACCOUNT_ID.r2.cloudflarestorage.com" \
     R2_ACCESS_KEY_ID="your_r2_access_key_id" \
     R2_SECRET_ACCESS_KEY="your_r2_secret_access_key" \
     R2_BUCKET_NAME="hangai-audio" \
     HF_TOKEN="hf_your_huggingface_token"
   ```
   Run this as ONE command in your terminal.

5. Deploy the app. First download the project files, then run:
   ```
   modal deploy modal_app.py
   ```
   Wait ~3-5 minutes. Modal will build the Docker image and install all dependencies.
   When it finishes, it prints two URLs like:
   ```
   ✓ Created get-presigned-url web function → https://USERNAME--hangai-get-presigned-url.modal.run
   ✓ Created trigger-processing web function → https://USERNAME--hangai-trigger-processing.modal.run
   ```
   Copy both URLs. You need them in the next step.

---

## STEP 5 — Configure and deploy the frontend

1. Open `index.html` in any text editor (Notepad, VS Code, anything)
2. Find the CONFIG section near the top (around line 20):
   ```javascript
   const CONFIG = {
     SUPABASE_URL:        "https://XXXXXXXXXX.supabase.co",
     SUPABASE_ANON_KEY:   "eyJhbGciOiJIUzI1NiIsInR5...",
     MODAL_PRESIGNED_URL: "https://USERNAME--hangai-get-presigned-url.modal.run",
     MODAL_TRIGGER_URL:   "https://USERNAME--hangai-trigger-processing.modal.run",
   };
   ```
3. Replace each value with your actual values from the steps above
4. Save the file

5. Now deploy to GitHub Pages:
   - Go to https://github.com → Create a new repository → name it `hangai` → Public
   - Click "uploading an existing file" → drag and drop `index.html` → commit
   - Go to repository Settings → Pages → Source → "Deploy from branch" → main → / (root) → Save
   - Wait 2 minutes. Your site will be at: `https://YOUR_GITHUB_USERNAME.github.io/hangai`

---

## STEP 6 — Test it works

1. Open `https://YOUR_GITHUB_USERNAME.github.io/hangai` on your laptop
2. Click "Start recording" → allow microphone → say a few sentences in Hindi and English → click stop
3. You should see a popup saying "X chunks uploaded"
4. Wait 5-10 minutes (shorter for test since it's a short recording)
5. Refresh the page — you should see your session with status changing from "processing" to "completed"
6. Click the session to see the summary, transcript, timeline, and insights

If it works, you're done.

---

## STEP 7 — Using it for real hangouts

Before leaving:
1. Open terminal on laptop
2. Run: `modal deploy modal_app.py`  ← this keeps your endpoints alive

At the hangout:
1. Open `https://YOUR_GITHUB_USERNAME.github.io/hangai` in Chrome on your phone
2. Tap "Start recording" → allow mic → type a session name → put phone in pocket
3. Keep Chrome tab open in background (important — closing tab stops recording)

When leaving:
1. Open Chrome tab again → tap "Stop recording" → wait for upload confirmation

At home:
1. Open dashboard on any device
2. Wait 20-30 min if not already done
3. Read everything

---

## What to change if something breaks

| Problem | What to change |
|---|---|
| "Microphone denied" | Chrome settings → allow mic for your GitHub Pages URL |
| Chunks not uploading | Check R2 CORS settings (Step 2, point 5) |
| Processing never starts | Check Modal dashboard → hangai app → logs |
| Diarization fails with 403 | You didn't accept HuggingFace model terms (Step 3, point 5) |
| Summaries are in wrong language | Edit the prompt in `modal_app.py` around line 170 |
| Want a different session name | Change the `prompt()` line in `index.html` around line 110 |
| Want longer summaries | Change `max_new_tokens=2000` to `max_new_tokens=3000` in `modal_app.py` |
| Want to use Qwen2.5-14B (better quality) | Change `"Qwen/Qwen2.5-7B-Instruct"` to `"Qwen/Qwen2.5-14B-Instruct"` in `modal_app.py` (needs more VRAM — change gpu="A10G" to gpu="A100") |

---

## API keys summary — where each one goes

| Key | Where to get it | Where it goes |
|---|---|---|
| SUPABASE_URL | Supabase → Settings → API | Modal secret + index.html CONFIG |
| SUPABASE_ANON_KEY | Supabase → Settings → API | index.html CONFIG only |
| SUPABASE_SERVICE_KEY | Supabase → Settings → API | Modal secret only (never in frontend!) |
| R2_ENDPOINT_URL | Cloudflare R2 → API Tokens | Modal secret only |
| R2_ACCESS_KEY_ID | Cloudflare R2 → API Tokens | Modal secret only |
| R2_SECRET_ACCESS_KEY | Cloudflare R2 → API Tokens | Modal secret only |
| R2_BUCKET_NAME | "hangai-audio" (you chose this) | Modal secret only |
| HF_TOKEN | HuggingFace → Settings → Tokens | Modal secret only |
| MODAL_PRESIGNED_URL | Modal deploy output | index.html CONFIG only |
| MODAL_TRIGGER_URL | Modal deploy output | index.html CONFIG only |

---

## Estimated costs per month (1 session/week)

- Supabase: $0
- GitHub Pages: $0
- Cloudflare R2: $0 (under 10GB free tier)
- Modal GPU: ~$1-4 total
- **Total: ~$1-4/month**
