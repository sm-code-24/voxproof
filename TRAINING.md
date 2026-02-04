# 🎓 VoxProof - Model Training Guide

Complete documentation for training the VoxProof voice detection model with real audio data.

---

## Overview

The VoxProof classifier distinguishes between **AI-generated** and **human** voices using:

- **Acoustic features** (MFCCs, pitch, spectral characteristics)
- **Deep embeddings** from wav2vec2 pre-trained model

This guide covers training with real-world audio data for production use.

---

## Quick Start

```bash
# Train with default synthetic data (demo only)
python train.py

# Train with real data (see Data Preparation section below)
python train.py --data-dir ./training_data --epochs 50
```

---

## Data Preparation

### Required Folder Structure

```
training_data/
├── human/
│   ├── speaker1_sample1.mp3
│   ├── speaker1_sample2.mp3
│   ├── speaker2_sample1.mp3
│   └── ...
└── ai_generated/
    ├── elevenlabs_sample1.mp3
    ├── bark_sample1.mp3
    ├── openai_tts_sample1.mp3
    └── ...
```

### Audio Requirements

| Requirement | Value                              |
| ----------- | ---------------------------------- |
| Format      | MP3, WAV, M4A, FLAC                |
| Duration    | 1-30 seconds per sample            |
| Sample Rate | Any (resampled to 16kHz)           |
| Channels    | Mono or Stereo (converted to mono) |
| Language    | Multi-language supported           |

### Recommended Dataset Size

| Dataset Size    | Accuracy | Use Case                 |
| --------------- | -------- | ------------------------ |
| 100 samples     | ~70%     | Testing/debugging        |
| 1,000 samples   | ~85%     | Proof of concept         |
| 10,000 samples  | ~92%     | Production-ready         |
| 50,000+ samples | ~97%     | High-accuracy production |

---

## Collecting Training Data

### Human Voice Sources

1. **Public Datasets:**
   - [LibriSpeech](https://www.openslr.org/12/) - 1000 hours of English audiobooks
   - [Common Voice](https://commonvoice.mozilla.org/) - Multi-language crowdsourced
   - [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) - Celebrity interviews
   - [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) - 110 English speakers

2. **Custom Recording:**
   - Record yourself and colleagues
   - Use phone/laptop microphone (natural noise is good!)
   - Vary speaking styles, emotions, environments

### AI-Generated Voice Sources

1. **TTS Services (create samples):**
   - [ElevenLabs](https://elevenlabs.io/) - High-quality neural TTS
   - [Bark](https://github.com/suno-ai/bark) - Open-source TTS
   - [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
   - [Google Cloud TTS](https://cloud.google.com/text-to-speech)
   - [Amazon Polly](https://aws.amazon.com/polly/)
   - [Microsoft Azure TTS](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/)

2. **Public AI Audio Datasets:**
   - [ASVspoof](https://www.asvspoof.org/) - Spoofing detection dataset
   - [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) - Deepfake celebrity

3. **Generating AI Samples:**

   ```python
   # Example: Generate samples with ElevenLabs API
   import requests

   ELEVENLABS_API_KEY = "your-key"

   text = "Hello, this is a test of synthetic speech generation."

   response = requests.post(
       "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
       headers={"xi-api-key": ELEVENLABS_API_KEY},
       json={"text": text, "model_id": "eleven_monolingual_v1"}
   )

   with open("ai_sample.mp3", "wb") as f:
       f.write(response.content)
   ```

---

## Training Script Usage

### Basic Training

```bash
python train.py
```

This uses synthetic data (for demo/testing only).

### Training with Real Data

Create a custom training script:

```python
# train_real_data.py
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import VoxProof modules
from model.model import VoiceClassifier, Wav2VecEmbedder
from audio.processing import AudioProcessor

class RealVoiceDataset(Dataset):
    """Dataset loader for real audio files."""

    def __init__(self, data_dir: str, cache_features: bool = True):
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor()
        self.embedder = Wav2VecEmbedder()
        self.cache_features = cache_features
        self.cache = {}

        # Collect all audio files
        self.samples = []

        # Human samples (label = 0)
        human_dir = self.data_dir / "human"
        if human_dir.exists():
            for audio_file in human_dir.glob("*"):
                if audio_file.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]:
                    self.samples.append((str(audio_file), 0))

        # AI-generated samples (label = 1)
        ai_dir = self.data_dir / "ai_generated"
        if ai_dir.exists():
            for audio_file in ai_dir.glob("*"):
                if audio_file.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]:
                    self.samples.append((str(audio_file), 1))

        print(f"Loaded {len(self.samples)} samples")
        print(f"  Human: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  AI: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        # Check cache
        if self.cache_features and audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            # Load and process audio
            try:
                waveform = self.audio_processor.load_audio_file(audio_path)
                acoustic_features = self.audio_processor.extract_features(waveform)
                embedding = self.embedder.extract_embedding(waveform)
                features = np.concatenate([acoustic_features, embedding])
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Return zeros if processing fails
                features = np.zeros(786, dtype=np.float32)

            if self.cache_features:
                self.cache[audio_path] = features

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def train_with_real_data(data_dir: str, epochs: int = 50, batch_size: int = 32):
    """Train model with real audio data."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = RealVoiceDataset(data_dir)

    # Split into train/val/test (70/15/15)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = VoiceClassifier(dropout_rate=0.3).to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_set)
        train_acc = train_correct / len(train_set)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_set)
        val_acc = val_correct / len(val_set)

        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model/classifier_best.pth")
            print("  ✓ Saved best model!")

    # Final test evaluation
    model.load_state_dict(torch.load("model/classifier_best.pth"))
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(features)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (preds == labels).sum().item()

    test_acc = test_correct / len(test_set)
    print(f"\n✓ Final Test Accuracy: {test_acc:.2%}")

    # Save final model
    torch.save(model.state_dict(), "model/classifier.pth")
    print("✓ Model saved to model/classifier.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./training_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_with_real_data(args.data_dir, args.epochs, args.batch_size)
```

### Run Training

```bash
# Prepare data
mkdir -p training_data/human training_data/ai_generated
# Copy audio files to respective folders...

# Train
python train_real_data.py --data-dir ./training_data --epochs 50
```

---

## Feature Extraction Details

### Acoustic Features (18 dimensions)

| Feature            | Dimensions | Description                                          |
| ------------------ | ---------- | ---------------------------------------------------- |
| MFCCs              | 13         | Mel-frequency cepstral coefficients (speech texture) |
| Pitch Mean         | 1          | Average fundamental frequency (F0)                   |
| Pitch Std          | 1          | Pitch variation (monotone vs expressive)             |
| Spectral Rolloff   | 1          | Frequency below which 85% of energy is contained     |
| Zero Crossing Rate | 1          | Measure of high-frequency content/noise              |
| Duration           | 1          | Audio length in seconds                              |

### wav2vec2 Embedding (768 dimensions)

- Pre-trained on 960 hours of English speech
- Captures high-level speech representations
- Mean-pooled across time dimension

---

## Hyperparameter Tuning

### Key Parameters

```python
class TrainingConfig:
    # Architecture
    INPUT_DIM = 786          # 18 acoustic + 768 embedding

    # Training
    BATCH_SIZE = 32          # Increase for faster training (if GPU memory allows)
    LEARNING_RATE = 0.001    # Start here, reduce if unstable
    WEIGHT_DECAY = 1e-4      # L2 regularization
    EPOCHS = 100             # With early stopping
    DROPOUT_RATE = 0.3       # Increase if overfitting

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
```

### Tips for Better Performance

1. **Balanced Dataset:** Equal human/AI samples prevents bias
2. **Diverse Sources:** Include multiple TTS systems and speakers
3. **Augmentation:** Add noise, time-stretch to improve robustness
4. **Cross-Validation:** Use k-fold for more reliable evaluation

---

## Data Augmentation

Improve model robustness with augmentation:

```python
import numpy as np
from scipy.signal import resample

def augment_audio(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply random augmentation to audio."""

    augmented = waveform.copy()

    # Random noise injection
    if np.random.random() > 0.5:
        noise = np.random.randn(len(augmented)) * 0.005
        augmented = augmented + noise

    # Random volume change
    if np.random.random() > 0.5:
        volume_factor = np.random.uniform(0.8, 1.2)
        augmented = augmented * volume_factor

    # Random time stretch (resampling trick)
    if np.random.random() > 0.5:
        stretch_factor = np.random.uniform(0.9, 1.1)
        new_length = int(len(augmented) * stretch_factor)
        augmented = resample(augmented, new_length)

    # Normalize
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-7)

    return augmented
```

---

## Model Evaluation

### Metrics to Track

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    # Calculate metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.2%}")
    print(f"Precision: {precision_score(all_labels, all_preds):.2%}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.2%}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds):.2%}")
    print(f"ROC-AUC:   {roc_auc_score(all_labels, all_probs):.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*50)
```

### Expected Results

| Metric   | Poor  | Acceptable | Good      | Excellent |
| -------- | ----- | ---------- | --------- | --------- |
| Accuracy | <70%  | 70-85%     | 85-95%    | >95%      |
| F1 Score | <0.7  | 0.7-0.85   | 0.85-0.95 | >0.95     |
| ROC-AUC  | <0.75 | 0.75-0.9   | 0.9-0.97  | >0.97     |

---

## Deploying Updated Model

After training, deploy your new model:

```bash
# Copy trained weights
cp model/classifier_best.pth model/classifier.pth

# Test locally
uvicorn app:app --reload

# Commit and push to trigger Railway deployment
git add model/classifier.pth
git commit -m "Update model with real training data"
git push origin main
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Disable feature caching
dataset = RealVoiceDataset(data_dir, cache_features=False)

# Use CPU for feature extraction, GPU for training
embedder = Wav2VecEmbedder(device="cpu")
```

### Training is Slow

```python
# Use GPU
device = torch.device("cuda")

# Increase workers
train_loader = DataLoader(train_set, batch_size=32, num_workers=4)

# Pre-extract features (one-time)
# Save features to .npy files and load them instead
```

### Poor Accuracy

1. **Check data balance:** Ensure 50/50 split
2. **Check data quality:** Remove corrupt files
3. **Increase diversity:** Add more varied samples
4. **Reduce dropout:** If underfitting
5. **Increase dropout:** If overfitting

---

## Advanced: Fine-tuning wav2vec2

For maximum accuracy, fine-tune wav2vec2 itself (requires more GPU memory):

```python
from transformers import Wav2Vec2Model, Wav2Vec2Config

class FineTunedVoxProof(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        # Freeze early layers (optional)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )

    def forward(self, waveform):
        # waveform: (batch, samples)
        outputs = self.wav2vec2(waveform)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return self.classifier(embedding)
```

---

## Next Steps

1. ✅ Collect diverse training data
2. ✅ Balance dataset (50% human, 50% AI)
3. ✅ Train with real data
4. ✅ Evaluate on held-out test set
5. ✅ Deploy updated model
6. ✅ Monitor production performance

---

**Need Help?**

- Check `train.py` for complete synthetic training example
- Review model architecture in `model/model.py`
- Test locally before deploying

Happy training! 🎓
