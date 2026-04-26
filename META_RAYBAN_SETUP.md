# Meta Ray-Ban Smart Glasses Setup Guide

## Overview

The continuous camera test (`test_camera_continuous.py`) **can work with Meta Ray-Bans**, but requires additional setup to stream video from the glasses to your laptop.

## Current Setup

### What Works Now ✅
- **Laptop Camera**: Direct testing with built-in webcam
- **USB Cameras**: Any USB-connected camera
- **IP Cameras**: RTSP/RTMP network streams

### What Needs Setup for Meta Ray-Bans 🔧
The Meta Ray-Ban glasses need to be mirrored to your laptop via:
1. **ADB (Android Debug Bridge)** - to connect to the glasses
2. **scrcpy** - to mirror the glasses display/camera
3. **RTMP/RTSP Server** - to stream the video feed

## Two Ways to Use the Demo

### Option 1: Quick Demo with Laptop Camera (Recommended for Testing)

This is what you just tested successfully:

```bash
python test_camera_continuous.py
```

**Pros:**
- ✅ Works immediately
- ✅ No additional hardware setup
- ✅ Tests all face recognition algorithms
- ✅ Validates the entire pipeline

**Cons:**
- ❌ Not first-person POV like the glasses
- ❌ Doesn't demonstrate the wearable aspect

### Option 2: Full Demo with Meta Ray-Bans (Production Setup)

This requires the complete hardware setup:

```bash
# After setting up the glasses stream (see below)
python test_camera_continuous.py --source rtsp://localhost:8554/live/stream
```

## Setting Up Meta Ray-Bans for Demo

### Prerequisites

1. **Meta Ray-Ban Smart Glasses** paired with your phone
2. **Meta View app** installed on your phone
3. **ADB tools** installed on your laptop
4. **scrcpy** installed on your laptop
5. **MediaMTX or similar RTSP server** (already in your project: `mediamtx.yml`)

### Step-by-Step Setup

#### 1. Enable Developer Mode on Glasses

The Meta Ray-Bans run on a Qualcomm platform. You need to enable developer access:

```bash
# Connect glasses via USB-C cable to laptop
adb devices
# Should show your glasses device
```

**Note:** Meta Ray-Bans may have limited ADB access. Check Meta's developer documentation for current capabilities.

#### 2. Install scrcpy

```bash
# macOS
brew install scrcpy

# Linux
sudo apt install scrcpy

# Windows
# Download from: https://github.com/Genymobile/scrcpy/releases
```

#### 3. Start RTSP Server

Your project already has MediaMTX configured:

```bash
# Start the RTSP server
./mediamtx

# Or if you have it as a service
brew services start mediamtx
```

#### 4. Mirror Glasses to RTSP Stream

```bash
# Option A: Direct scrcpy mirror (if glasses support it)
scrcpy --video-source=camera --camera-id=0 --no-audio

# Option B: Use ffmpeg to capture and stream
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast \
  -f rtsp rtsp://localhost:8554/live/stream
```

#### 5. Run the Continuous Test with Stream

```bash
python test_camera_continuous.py --source rtsp://localhost:8554/live/stream
```

### Alternative: Phone Camera as Proxy

If direct glasses access is limited, you can use your phone as a proxy:

1. **Install IP Webcam app** on your phone
2. **Mount phone on glasses** or hold it to simulate POV
3. **Stream from phone**:

```bash
# Phone will provide an RTSP URL like:
python test_camera_continuous.py --source rtsp://192.168.1.100:8080/video
```

## Current Project Integration

### Vision Engine Already Supports Streams

The main vision engine (`services/vision/face_recognition_engine.py`) already supports:

```python
# From .env or command line
RTMP_STREAM_URL=rtmp://localhost/live/stream
# or
RTSP_STREAM_URL=rtsp://localhost:8554/live/stream
```

### Running Full System with Glasses

```bash
# 1. Start RTSP server
./mediamtx

# 2. Set up glasses stream (see steps above)

# 3. Configure .env
echo "RTMP_STREAM_URL=rtsp://localhost:8554/live/stream" >> .env

# 4. Launch all services
python run_all.py
```

## Testing Strategy for Demo

### Recommended Approach

1. **Test with laptop camera first** (what you just did) ✅
   - Validates face recognition works
   - Tests all algorithms
   - Confirms known faces are loaded correctly

2. **Set up glasses stream** (if time permits)
   - Follow setup steps above
   - Test stream connectivity
   - Verify video quality

3. **Demo with whichever works best**
   - Laptop camera: "Here's the face recognition working in real-time"
   - Glasses stream: "This is the actual first-person POV from the glasses"

## Troubleshooting

### Glasses Not Detected by ADB

```bash
# Check USB connection
adb devices

# If not showing, try:
adb kill-server
adb start-server
adb devices
```

### Stream Not Connecting

```bash
# Test RTSP server
ffprobe rtsp://localhost:8554/live/stream

# Check if MediaMTX is running
ps aux | grep mediamtx
```

### Low Frame Rate

```bash
# Use lower resolution for streaming
python test_camera_continuous.py --source rtsp://localhost:8554/live/stream
# The script will automatically handle frame rate
```

## What the Continuous Test Shows

Regardless of video source (laptop or glasses), the test demonstrates:

- ✅ **Real-time face detection** with bounding boxes
- ✅ **Face recognition** matching against known profiles
- ✅ **Confidence scores** for each match
- ✅ **Processing speed** (FPS and ms per frame)
- ✅ **Live overlay** with person names
- ✅ **Snapshot capability** for documentation

## Audio Output to Glasses

The audio output is already configured in `.env`:

```bash
GLASSES_AUDIO_DEVICE=MacBook Air Speakers
```

To route audio to the actual glasses speaker:

1. **Pair glasses via Bluetooth** to your laptop
2. **Find the audio device name**:
   ```bash
   python -c "import pygame; pygame.mixer.init(); print(pygame.mixer.get_init())"
   ```
3. **Update .env**:
   ```bash
   GLASSES_AUDIO_DEVICE="Meta Ray-Ban Audio"
   ```

## Summary

### For Your Demo:

**Quick Path (Recommended):**
- ✅ Use laptop camera (already working)
- ✅ Show face recognition in real-time
- ✅ Demonstrate all features
- 📝 Explain: "In production, this runs on Meta Ray-Bans with first-person POV"

**Full Path (If Time Permits):**
- 🔧 Set up glasses streaming
- 🔧 Test RTSP connection
- ✅ Run with actual glasses feed
- ✅ Show true first-person perspective

Both approaches demonstrate the same AI capabilities. The laptop camera is faster to set up and equally impressive for showing the face recognition technology.
