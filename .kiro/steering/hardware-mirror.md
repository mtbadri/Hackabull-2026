---
inclusion: manual
---

# AuraGuard AI — Hardware Mirror Setup (Meta Smart Glasses → Laptop)

This guide covers how to mirror the Meta Smart Glasses POV stream to the laptop so the Vision Engine can read it as a video source.

## Prerequisites

- Meta Smart Glasses paired and connected to the laptop via USB or Bluetooth
- ADB (Android Debug Bridge) installed: `brew install android-platform-tools`
- scrcpy installed: `brew install scrcpy`

## Step 1 — Enable Developer Mode on the Glasses

1. On the glasses companion app, navigate to **Settings → About → Software Version**
2. Tap the version number 7 times to unlock Developer Mode
3. Enable **USB Debugging** in the Developer Options menu

## Step 2 — Connect via ADB

```bash
# Verify the glasses are detected
adb devices

# Expected output:
# List of devices attached
# XXXXXXXXXXXXXXXX    device
```

If the device shows as `unauthorized`, accept the RSA key prompt on the glasses.

## Step 3 — Mirror the Display with scrcpy

```bash
# Mirror with a fixed window title (Vision Engine reads this window)
scrcpy --window-title "MetaGlassesMirror" --no-audio

# Or mirror to a specific display position for the demo layout
scrcpy --window-title "MetaGlassesMirror" --window-x 0 --window-y 0 --window-width 960 --window-height 540
```

Leave this window open. The Vision Engine's `VideoCapture` will read from it.

## Step 4 — Configure the Vision Engine Video Source

In `.env`, set `VIDEO_SOURCE` to the window capture index or the scrcpy virtual device:

```bash
# Use default webcam (index 0) for testing without glasses
VIDEO_SOURCE=0

# Use the scrcpy mirror window (index may vary; try 1 or 2 if 0 is the webcam)
VIDEO_SOURCE=1
```

To find the correct index, run:
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Index {i}: available')
        cap.release()
"
```

## Step 5 — Verify the Mirror is Working

```bash
python -c "
import cv2
cap = cv2.VideoCapture(1)  # replace with your VIDEO_SOURCE index
ret, frame = cap.read()
print('Frame captured:', ret, frame.shape if ret else 'N/A')
cap.release()
"
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `adb devices` shows nothing | Try a different USB cable; ensure USB Debugging is enabled |
| scrcpy window is black | Glasses may be in sleep mode; tap the touchpad to wake them |
| Vision Engine reads wrong camera | Increment `VIDEO_SOURCE` in `.env` until the correct feed appears |
| Audio not routing to glasses | Verify `GLASSES_AUDIO_DEVICE` in `.env` matches the exact device name shown in System Preferences → Sound |

## Finding the Glasses Audio Device Name

```bash
python -c "
import pygame
pygame.mixer.init()
# On macOS, list audio devices via sounddevice
import sounddevice as sd
print(sd.query_devices())
"
```

Look for a device name containing "Meta" or "Ray-Ban" and set `GLASSES_AUDIO_DEVICE` to that exact string.

## Demo Layout Recommendation

For the hackathon demo, arrange windows on the laptop screen:
- **Left half**: scrcpy mirror (patient POV)
- **Right top**: Terminal showing Brain logs (events firing in real time)
- **Right bottom**: Browser at `http://localhost:8501` (Caregiver Portal)

This lets judges see the patient's perspective, the system processing events, and the caregiver dashboard simultaneously.
