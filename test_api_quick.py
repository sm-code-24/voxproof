"""
Quick API Test Script
Test the VoxProof API with a simple example
"""

import requests
import base64
import json

# API Configuration
API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "voxproof-secret-key-2024"

def test_health():
    """Test the health endpoint"""
    response = requests.get("http://localhost:8000/health")
    print("🔍 Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_voice_detection(audio_file_path):
    """Test voice detection with an audio file"""
    # Read and encode audio file
    try:
        with open(audio_file_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        # Prepare request
        payload = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
        
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        print(f"📤 Sending request for: {audio_file_path}")
        print(f"   Audio size: {len(audio_base64)} bytes (base64)")
        print()
        
        # Make request
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection Result:")
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidenceScore']:.1%}")
            print(f"   Explanation: {result['explanation']}")
        else:
            print(f"❌ Error {response.status_code}:")
            print(json.dumps(response.json(), indent=2))
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("VoxProof API - Quick Test")
    print("=" * 70)
    print()
    
    # Test health endpoint
    test_health()
    
    # Test with audio file (update path to your audio file)
    audio_file = input("Enter path to MP3 file (or press Enter to skip): ").strip()
    
    if audio_file:
        if audio_file.startswith('"') and audio_file.endswith('"'):
            audio_file = audio_file[1:-1]  # Remove quotes
        test_voice_detection(audio_file)
    else:
        print("ℹ️  Skipping audio test - no file provided")
        print()
        print("To test with an audio file, run:")
        print('  python test_api_quick.py')
        print("  and enter the path to your MP3 file when prompted")
    
    print()
    print("=" * 70)
    print("📚 API Documentation available at: http://localhost:8000/docs")
    print("=" * 70)
