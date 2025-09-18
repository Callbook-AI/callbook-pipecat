#!/bin/bash
# Setup script for AssemblyAI Real-time Transcription Test

echo "🚀 Setting up AssemblyAI Real-time Transcription Test..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install system dependencies for PyAudio (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - checking for Homebrew..."
    if command -v brew &> /dev/null; then
        echo "🍺 Installing PortAudio via Homebrew..."
        brew install portaudio
    else
        echo "⚠️  Homebrew not found. Please install PortAudio manually:"
        echo "   Visit: https://brew.sh/ to install Homebrew"
        echo "   Then run: brew install portaudio"
    fi
fi

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python requirements
echo "📥 Installing Python dependencies..."
pip install -r test_requirements.txt

echo ""
echo "🎉 Setup complete!"
echo "=================================================="
echo ""
echo "🎤 To run the AssemblyAI test:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the test: python test_assembly_realtime.py"
echo ""
echo "📝 The test will:"
echo "   • Connect to AssemblyAI in Spanish mode"
echo "   • Capture audio from your microphone"
echo "   • Show real-time transcription with metrics"
echo "   • Display performance statistics"
echo ""
echo "🛑 Press Ctrl+C to stop the test"
echo "=================================================="
