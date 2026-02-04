# 🎙️ VoxProof - AI Voice Detection API

> **Hackathon Submission:** AI for Fraud Detection Challenge  
> **Team:** VoxProof  
> **Live Demo:** [voxproof.up.railway.app](https://voxproof.up.railway.app)

An intelligent API system that detects whether a voice sample is AI-generated (deepfake) or spoken by a real human. Designed to combat voice-based fraud, identity theft, and misinformation.

---

## 🎯 Problem Statement

With the rise of AI voice synthesis tools (ElevenLabs, Bark, OpenAI TTS), criminals can now clone anyone's voice in minutes. This technology enables:

- **Voice phishing (vishing)** scams impersonating family members or executives
- **Identity fraud** bypassing voice-based authentication systems
- **Misinformation** through fake audio recordings of public figures

**VoxProof** provides an API to detect AI-generated voices and protect against these threats.

---

## 🚀 Quick Start

### One-Click Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/voxproof)

Or deploy manually:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/VoxProof.git
cd VoxProof

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
winget install Gyan.FFmpeg  # Windows
# sudo apt install ffmpeg   # Linux
# brew install ffmpeg       # Mac

# Run server
uvicorn app:app --reload --port 8000
```

Visit: http://localhost:8000/docs for API documentation.

---

## 🏗️ Architecture

```
VoxProof/
├── app.py                  # FastAPI entry point
├── train.py                # Model training script
├── test_client.py          # CLI testing tool
├── audio/
│   └── processing.py       # Audio decoding & feature extraction
├── model/
│   ├── model.py            # Neural network + wav2vec2 embeddings
│   ├── classifier.pth      # Trained model weights
│   └── classifier_best.pth # Best checkpoint
├── utils/
│   └── explain.py          # Human-readable explanations
├── railway.json            # Railway deployment config
├── requirements.txt        # Python dependencies
├── DEPLOYMENT.md           # Railway deployment guide
└── TRAINING.md             # Model training documentation
```

---

## 🔌 API Reference

### Health Check

```http
GET /health
```

Returns: `{"status": "ok", "model_loaded": true}`

### Voice Detection

```http
POST /api/voice-detection
Headers:
  x-api-key: <API_KEY>
  Content-Type: application/json

Body:
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Response:**

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Unnaturally stable pitch pattern detected - synthetic voices often lack natural pitch fluctuations"
}
```

**Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu

---

## 🧠 How It Works

### 1. Audio Processing Pipeline

- Base64 decode → MP3 decode → Resample to 16kHz mono → Normalize

### 2. Feature Extraction

| Feature            | Description                         |
| ------------------ | ----------------------------------- |
| MFCCs (13)         | Mel-frequency cepstral coefficients |
| Pitch Stats        | Mean and standard deviation of F0   |
| Spectral Rolloff   | Frequency energy distribution       |
| Zero Crossing Rate | Signal noise characteristics        |

### 3. Deep Embeddings

- **Model:** `facebook/wav2vec2-base-960h`
- **Output:** 768-dimensional speech representation
- Pre-trained on 960 hours of LibriSpeech

### 4. Classification

```
Input: 786 features (18 acoustic + 768 embedding)
    ↓
4 Hidden Layers (BatchNorm + ReLU + Dropout)
    ↓
Sigmoid Output → AI_GENERATED (1) or HUMAN (0)
```

### 5. Explainable AI

Rule-based explanations analyzing:

- Pitch stability (AI voices are unnaturally stable)
- Signal clarity (synthetic audio lacks natural noise)
- Spectral patterns (TTS has characteristic frequency distribution)

---

## 🧪 Testing

### Using cURL

```bash
# Encode audio
base64 audio.mp3 > audio_b64.txt

# Test API
curl -X POST "https://your-app.up.railway.app/api/voice-detection" \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"language":"English","audioFormat":"mp3","audioBase64":"<base64>"}'
```

### Using Python

```python
import base64, requests

with open("audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://your-app.up.railway.app/api/voice-detection",
    headers={"x-api-key": "your-api-key"},
    json={"language": "English", "audioFormat": "mp3", "audioBase64": audio_b64}
)
print(response.json())
```

### Using Test Client

```bash
python test_client.py path/to/audio.mp3
```

---

## ⚙️ Configuration

### Environment Variables

| Variable        | Default                       | Description                  |
| --------------- | ----------------------------- | ---------------------------- |
| `API_KEY`       | `voxproof-secret-key-2024`    | API authentication key       |
| `MODEL_PATH`    | `model/classifier.pth`        | Path to trained model        |
| `WAV2VEC_MODEL` | `facebook/wav2vec2-base-960h` | HuggingFace model ID         |
| `SAMPLE_RATE`   | `16000`                       | Audio sample rate            |
| `PORT`          | `8000`                        | Server port (set by Railway) |

---

## 📊 Performance

| Metric         | Value                          |
| -------------- | ------------------------------ |
| Inference Time | ~500ms (CPU)                   |
| Model Size     | ~5MB (classifier)              |
| wav2vec2 Size  | ~360MB (downloaded on startup) |
| Memory Usage   | ~2GB RAM                       |
| Accuracy\*     | 95%+ (on synthetic data)       |

\*Note: Accuracy on real-world data depends on training dataset quality.

---

## 🔒 Security Features

- ✅ API key authentication
- ✅ Input validation with Pydantic
- ✅ CORS middleware
- ✅ Request size limits
- ✅ Health check endpoints

---

## 📁 Project Files

| File                             | Purpose                     |
| -------------------------------- | --------------------------- |
| [app.py](app.py)                 | Main FastAPI application    |
| [train.py](train.py)             | Model training script       |
| [TRAINING.md](TRAINING.md)       | How to train with real data |
| [DEPLOYMENT.md](DEPLOYMENT.md)   | Railway deployment guide    |
| [test_client.py](test_client.py) | CLI testing tool            |

---

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete Railway deployment instructions.

### Quick Deploy Steps:

1. Push code to GitHub
2. Connect repository to Railway
3. Set environment variables
4. Deploy!

Railway auto-detects Python and configures everything using `railway.json`.

---

## 📚 Training Your Own Model

See [TRAINING.md](TRAINING.md) for comprehensive training documentation.

The included model weights are trained on synthetic data for demonstration. For production use:

1. Collect labeled audio samples (AI-generated vs human)
2. Organize in folder structure
3. Run training script
4. Deploy updated model

---

## 🛠️ Technology Stack

- **Backend:** FastAPI, Uvicorn
- **ML Framework:** PyTorch
- **Audio Processing:** librosa, pydub, FFmpeg
- **Speech Embeddings:** HuggingFace Transformers (wav2vec2)
- **Deployment:** Railway
- **Language:** Python 3.11

---

## 🐛 Troubleshooting

| Issue               | Solution                                   |
| ------------------- | ------------------------------------------ |
| FFmpeg not found    | Install FFmpeg and add to PATH             |
| Model download slow | First run downloads 360MB wav2vec2 model   |
| Out of memory       | Use CPU-only or increase Railway RAM       |
| Invalid audio       | Ensure MP3 is valid before Base64 encoding |

---

## 👥 Team

**VoxProof Team** - AI for Fraud Detection Hackathon 2026

---

## 📄 License

MIT License - Free to use and modify.

---

<div align="center">

**Built with ❤️ for AI for Fraud Detection Hackathon**

[Live Demo](https://voxproof.up.railway.app) • [API Docs](https://voxproof.up.railway.app/docs) • [GitHub](https://github.com/yourusername/VoxProof)

</div>
