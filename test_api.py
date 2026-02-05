"""
VoxProof API Test Client
========================

A comprehensive test client for the VoxProof voice detection API.
Supports both interactive mode and command-line usage.

Usage:
    Interactive:  python test_api.py
    CLI:          python test_api.py <audio_file.mp3> [language]
    
Environment Variables:
    API_KEY     - Required. Your API key for authentication.
    API_URL     - Optional. API base URL (default: http://localhost:8000)
    OUTPUT_DIR  - Optional. Directory for saving results (default: results)
"""

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    print("ERROR: API_KEY environment variable is required.")
    print("Please set it in your .env file or environment.")
    sys.exit(1)

API_URL = os.getenv("API_URL", "http://localhost:8000")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


# =============================================================================
# API Functions
# =============================================================================

def test_health(api_url: str = None, verbose: bool = True) -> bool:
    """
    Test the API health endpoint.
    
    Args:
        api_url: Base URL of the API
        verbose: Whether to print results
        
    Returns:
        True if healthy, False otherwise
    """
    if api_url is None:
        api_url = API_URL
        
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        health_data = response.json()
        
        if verbose:
            print("Health Check:")
            print(f"  Status: {health_data.get('status', 'unknown')}")
            print(f"  Models Loaded: {health_data.get('models_loaded', 'unknown')}")
            print(f"  Sample Rate: {health_data.get('sample_rate', 'unknown')} Hz")
            print()
        return True
        
    except requests.exceptions.ConnectionError:
        if verbose:
            print("ERROR: Could not connect to API.")
            print(f"  Is the server running at {api_url}?")
            print("  Start it with: uvicorn app:app --reload")
        return False
    except Exception as e:
        if verbose:
            print(f"ERROR: Health check failed: {e}")
        return False


def test_voice_detection(
    audio_path: str,
    language: str = "English",
    api_url: str = None,
    api_key: str = None,
    save_results: bool = True,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Test voice detection on an audio file.
    
    Args:
        audio_path: Path to the MP3 audio file
        language: Language of the audio
        api_url: Base URL of the API
        api_key: API key for authentication
        save_results: Whether to save results to JSON file
        verbose: Whether to print detailed output
        
    Returns:
        API response dict or None if failed
    """
    if api_url is None:
        api_url = API_URL
    if api_key is None:
        api_key = API_KEY
    
    # Clean up path (remove quotes if present)
    audio_path = audio_path.strip()
    if audio_path.startswith('"') and audio_path.endswith('"'):
        audio_path = audio_path[1:-1]
    if audio_path.startswith("'") and audio_path.endswith("'"):
        audio_path = audio_path[1:-1]
    
    # Validate file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"ERROR: File not found: {audio_path}")
        return None
    
    # Warn if not MP3
    if audio_file.suffix.lower() != ".mp3":
        print(f"WARNING: File may not be MP3 format: {audio_file.suffix}")
    
    # Read and encode audio
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Save base64 to file for debugging
        base64_file = Path("audio_base64.txt")
        with open(base64_file, "w") as f:
            f.write(audio_base64)
        if verbose:
            print(f"Base64 saved to: {base64_file.absolute()}")
        
        if verbose:
            print(f"Audio File: {audio_file.name}")
            print(f"  Size: {len(audio_bytes):,} bytes")
            print(f"  Base64: {len(audio_base64):,} chars")
            print()
            
    except Exception as e:
        print(f"ERROR: Could not read audio file: {e}")
        return None
    
    # Prepare request
    endpoint = f"{api_url}/api/voice-detection"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Make request
    if verbose:
        print(f"Sending request to {api_url}...")
        
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            if verbose:
                print()
                print("=" * 60)
                print("DETECTION RESULT")
                print("=" * 60)
                print(f"  Classification: {result.get('classification', 'N/A')}")
                print(f"  Confidence:     {result.get('confidenceScore', 0):.1%}")
                print(f"  Explanation:    {result.get('explanation', 'N/A')}")
                print("=" * 60)
            
            # Save results
            if save_results:
                _save_result(audio_path, language, len(audio_bytes), result, verbose)
            
            return result
            
        elif response.status_code == 401:
            print("ERROR: Invalid API key")
            print("  Check your API_KEY in .env file")
            return None
        else:
            print(f"ERROR: API returned status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"  Detail: {error_detail.get('detail', response.text)}")
            except:
                print(f"  Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        print("  The audio file may be too large or the server is busy")
        return None
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API")
        print(f"  Is the server running at {api_url}?")
        return None
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        return None


def _save_result(
    audio_path: str,
    language: str,
    audio_size: int,
    result: dict,
    verbose: bool = True
):
    """Save detection result to JSON file."""
    try:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "results.json"
        
        # Prepare entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "audio_file": Path(audio_path).name,
                "audio_size_bytes": audio_size,
                "language": language
            },
            "result": result
        }
        
        # Load existing or create new
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            # Handle case where results.json is a dict (from training script)
            if isinstance(all_results, dict):
                all_results = [all_results]
        else:
            all_results = []
        
        # Append and save
        all_results.append(entry)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"\nResults saved to: {output_file}")
            print(f"  Total entries: {len(all_results)}")
            
    except Exception as e:
        if verbose:
            print(f"WARNING: Could not save results: {e}")


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive():
    """Run in interactive mode."""
    print()
    print("=" * 60)
    print("VoxProof API - Interactive Test")
    print("=" * 60)
    print()
    
    # Health check
    print("Checking API health...")
    if not test_health(verbose=True):
        return
    
    # Get audio file
    audio_path = input("Enter path to MP3 file (or press Enter to skip): ").strip()
    
    if not audio_path:
        print()
        print("No file provided. Skipping voice detection test.")
        print()
        print("To test with an audio file, run again and provide a path.")
        return
    
    # Get language (optional)
    print(f"\nSupported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    language = input("Enter language (default: English): ").strip()
    if not language:
        language = "English"
    
    print()
    
    # Run detection
    test_voice_detection(audio_path, language)
    
    print()
    print("=" * 60)
    print(f"API Documentation: {API_URL}/docs")
    print("=" * 60)


# =============================================================================
# CLI Mode
# =============================================================================

def run_cli():
    """Run in CLI mode with command-line arguments."""
    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "English"
    
    print()
    
    # Health check
    print("Checking API health...")
    if not test_health(verbose=False):
        print("ERROR: API is not available")
        print("  Start the server: uvicorn app:app --reload")
        sys.exit(1)
    print("  OK")
    print()
    
    # Run detection
    result = test_voice_detection(audio_path, language)
    
    if result is None:
        sys.exit(1)
    print()


def print_usage():
    """Print usage information."""
    print()
    print("VoxProof API Test Client")
    print("=" * 40)
    print()
    print("Usage:")
    print("  Interactive: python test_api.py")
    print("  CLI:         python test_api.py <audio.mp3> [language]")
    print()
    print("Languages: Tamil, English, Hindi, Malayalam, Telugu")
    print()
    print("Configuration (from .env or environment):")
    print(f"  API_KEY:    {'*' * 8}...{API_KEY[-4:]}")
    print(f"  API_URL:    {API_URL}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print()
    print("Examples:")
    print("  python test_api.py")
    print("  python test_api.py sample.mp3")
    print("  python test_api.py sample.mp3 Hindi")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # Interactive mode
        run_interactive()
    elif len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help"):
        print_usage()
    elif len(sys.argv) >= 2:
        # CLI mode
        run_cli()
    else:
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
