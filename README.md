# VoxProof - AI Voice Detection API

> **Detect AI-generated voices in real-time** - Built for the AI for Fraud Detection Hackathon

[![Live API](https://img.shields.io/badge/Live%20API-Railway-blueviolet)](https://voxproof.up.railway.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)

---

## 🌐 Live Demo

**API Endpoint:** `https://voxproof.up.railway.app/api/voice-detection`

**API Documentation:** [https://voxproof.up.railway.app/docs](https://voxproof.up.railway.app/docs)

**Health Check:** [https://voxproof.up.railway.app/health](https://voxproof.up.railway.app/health)

---

## 🎯 The Problem

AI voice cloning tools make it easy to impersonate anyone. This enables:

- Voice phishing scams
- Identity fraud
- Fake audio misinformation

VoxProof provides an API to detect synthetic voices.

---

## 🔬 How It Works

```
Audio → Preprocessing → Feature Extraction → Neural Network → AI or Human?
```

**Features extracted:**

- **Acoustic features:** MFCCs (13 coefficients), pitch variance, spectral rolloff, zero-crossing rate
- **Deep embeddings:** Wav2Vec2 pretrained speech representations (768-dim)

**Model architecture:**

- Custom neural network classifier: `786 → 512 → 256 → 128 → 64 → 1`
- Binary classification with sigmoid output
- Trained on AI-generated vs human voice samples

---

## 📁 Project Structure

```
VoxProof/
├── app.py                 # FastAPI server with production logging
├── train_classifier.py    # Model training pipeline
├── audio/
│   └── processing.py      # Audio preprocessing & feature extraction
├── model/
│   ├── model.py           # Neural network architecture
│   └── classifier.pth     # Trained weights (PyTorch)
├── utils/
│   └── explain.py         # Human-readable explanation generator
├── Dockerfile             # Container configuration
├── railway.json           # Railway deployment config
└── requirements.txt       # Python dependencies
```

---

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/sm-code-24/VoxProof.git
cd VoxProof

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for audio processing)
# Windows: winget install Gyan.FFmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg

# Set environment variables
cp .env.example .env  # Edit with your API_KEY

# Run the API
uvicorn app:app --reload --port 8000
```

**Local API docs:** http://localhost:8000/docs

---

## 📡 API Usage

### Endpoint: POST /api/voice-detection

**Live URL:** `https://voxproof.up.railway.app/api/voice-detection`

**Headers:**

```
x-api-key: your-api-key
Content-Type: application/json
```

**Request Body:**

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Supported Languages:** English, Tamil, Hindi, Malayalam, Telugu

_Accepts flexible formats:_ `en`, `eng`, `english`, `English` (and similar for other languages like `ta`, `hi`, `ml`, `te`)

**Response (Success):**

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Unnaturally stable pitch detected"
}
```

---

## 🧪 Testing the API

### Quick Test with cURL (Live API)

```bash
# Encode your audio file
BASE64=$(base64 -w 0 your_audio.mp3)

# Test the live API
curl -X POST "https://voxproof.up.railway.app/api/voice-detection" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

### 1. Convert Audio to Base64

First, you need to encode your MP3 file to Base64:

**Windows (PowerShell):**

```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path/to/audio.mp3"))
$base64 | Out-File -Encoding utf8 audio_base64.txt
```

**Linux/Mac:**

```bash
base64 audio.mp3 > audio_base64.txt
# Remove newlines (optional, but recommended)
base64 -w 0 audio.mp3 > audio_base64.txt
```

This creates a text file with the Base64-encoded audio. Use the content for API requests.

### 2. Test with cURL

```bash
# Load Base64 from file and API key from .env
$BASE64 = Get-Content audio_base64.txt -Raw
$APIKEY = (Get-Content .env | Select-String 'API_KEY=' | ForEach-Object { $_ -replace 'API_KEY=', '' })

# Make request
curl -X POST "http://localhost:8000/api/voice-detection" `
  -H "x-api-key: $APIKEY" `
  -H "Content-Type: application/json" `
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

**Linux/Mac:**

```bash
# Load BASE64 and API key from .env
BASE64=$(cat audio_base64.txt)
API_KEY=$(grep '^API_KEY=' .env | cut -d'=' -f2)

curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

### 3. Test with Python

```python
import base64
import requests

# Read and encode audio file
with open("path/to/audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Load API key from environment
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Send request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "x-api-key": api_key,
        "Content-Type": "application/json"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
)

# Print response
print(response.json())
```

### 4. Test with Swagger UI

The easiest way - just visit the interactive API docs:

1. Start the server: `uvicorn app:app --reload --port 8000`
2. Open browser: http://localhost:8000/docs
3. Click "Try it out" on the `/api/voice-detection` endpoint
4. Paste your Base64-encoded audio into the `audioBase64` field
5. Click "Execute"

---

## 🏋️ Training

```bash
# Prepare dataset structure
dataset/
  human/  # Real voice samples (.mp3)
  ai/     # AI-generated samples (.mp3)

# Train the classifier
python train_classifier.py
```

**Output:** `model/classifier.pth`

**Training Features:**

- Automatic train/validation split
- Early stopping to prevent overfitting
- Checkpoints saved for best model

---

## ⚙️ Environment Variables

| Variable        | Default                       | Description                           |
| --------------- | ----------------------------- | ------------------------------------- |
| `API_KEY`       | (required)                    | API key for authentication            |
| `PORT`          | `8000`                        | Server port                           |
| `MODEL_PATH`    | `model/classifier.pth`        | Path to trained model weights         |
| `WAV2VEC_MODEL` | `facebook/wav2vec2-base-960h` | Wav2Vec2 model for feature extraction |
| `SAMPLE_RATE`   | `16000`                       | Audio sample rate in Hz               |

---

## 🛠️ Tech Stack

| Component         | Technology                          |
| ----------------- | ----------------------------------- |
| **API Framework** | FastAPI + Uvicorn                   |
| **ML Framework**  | PyTorch 2.0                         |
| **Speech Model**  | HuggingFace Transformers (Wav2Vec2) |
| **Audio**         | librosa + pydub + FFmpeg            |
| **Deployment**    | Railway (Docker container)          |

---

## 🚀 Deployment

Deployed on [Railway](https://railway.app) with automatic deployments from GitHub.

```bash
# Railway CLI deployment
railway up
```

The API automatically:

- Loads models on startup
- Structured JSON logging for monitoring
- Health checks at `/health`
- Auto-scales based on traffic

---

## 👥 Team

Built for the **AI for Fraud Detection Hackathon**

---

## 📄 License

MIT
