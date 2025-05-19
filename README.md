# EyeDetector—COM31006 Watermarking Project

**Author:** Aryan Golbaghi  
**Module:** COM31006 — Image-to-Image Steganography for Watermark Creation
**Purpose:** This project demonstrates image watermark embedding, verification, tamper detection, and recovery using non-overlapping SIFT keypoints and altering the least significant bit (LSBs) steganography.

---

## Overview

EyeDetector allows users to:
- Embed a watermark into a carrier image.
- Verify the authenticity of a suspected image.
- Detect and visualize mismatched or tampered watermark regions.
- Recover the original watermark using majority voting.

The watermark is embedded in the **(LSBs)** of the **blue channel** at **non-overlapping SIFT keypoints**, providing watermarking embedding, verification and temper detection.

---

## Features

- **Watermark Embedding** at strongest SIFT keypoints.
- **Watermark Verification** using geometric homography and bit-pattern similarity.
- **Tamper Detection** overlays the tempered points with red circle and generate mismatch analysis.
- **Watermark Recovery** recovers the water mark using majority technique voting.
- **Simple GUI** with image preview and progress feedback.

---

## How It Works

### Embedding Process
- Convert watermark image to 9x9 binary.
- Detect strong SIFT keypoints in carrier image.
- For each selected keypoint:
  - Embed the watermark in a square patch (centered on the keypoint) into the **LSB of the blue channel**.
- Save:
  - Modified image
  - Metadata (`.json`) including SIFT points(x,y positions, size and angles) and watermark segment.

### Verification
- Re-detect SIFT keypoints from suspect image.
- Match against original keypoints using spatial proximity.
- Verify bit similarity and compute geometric preservation (inlier ratio).

### Detection
- Visually mark mismatched regions with red circles on the suspect image.

### Recovery
- Extract embedded segments from all matched keypoints.
- Reconstruct watermark using majority voting.
- Optionally can upscale the result for visualization.

---

## GUI Preview

The interface provides buttons for:
- `Embed`: Add watermark to image.
- `Verify`: Check image authenticity.
- `Detect`: Highlight tampered regions.
- `Recover`: Reconstruct the watermark.

The GUI also includes progress feedback and real-time image previews.

---

## Requirements

Install dependencies (opencv-python, numpy, Pillow):
```bash
pip install -r requirements.txt
