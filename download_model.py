#!/usr/bin/env python3
"""Download SFace model from OpenCV GitHub."""

import urllib.request
import sys

url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
output = "tests/vision/models/face_recognition_sface_2021dec.onnx"

print(f"Downloading from: {url}")
print(f"Saving to: {output}")

try:
    urllib.request.urlretrieve(url, output)
    import os
    size = os.path.getsize(output)
    print(f"Downloaded! File size: {size} bytes")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)