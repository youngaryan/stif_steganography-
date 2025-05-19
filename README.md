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
```

---
## Project File Structure

```bash
stif_steganography-/
├── script/
│   ├── eye.py  #brain of the program
├── images/  #sample carrier and watermark image
│   ├── che.png
│   ├── watermark.png
├── res/  #folder which the genrated data(images and metadata) will be stored at
├── requirements.txt
├── .gitignore
└── README.md
```

---
## Example Usage
Inside your tereminal run:
```bash
python eye.py
```

Once ran you will see a simple GUI which from there using provided buttons you can embed a watermark inside a carrier image, verify, detect temperments, or recover a watermark.

![image](https://github.com/user-attachments/assets/9beac24a-33f1-4592-b292-3ae1665b751c)


**Embed Watermark:**

1. Select a carrier image (`.png` or `.tiff`).
2. Select a binary watermark image.
3. The program will generate:
   - A modified image: `res/<name>_modified.png`
   - A metadata file: `res/<name>_meta.json`

**Verify Image:**
1. Select a suspected image (`.png` or `.tiff`).
2. Select the metadata file
    - If the image is verified the program return autnticated.
    - Otherwise will return Tempered and the Inliers value [0,1].

<p float="left">
  <img src="https://github.com/user-attachments/assets/f4d859b1-d89c-4119-b1d4-c6005119a380" width="300" height="200" />
  <img src="https://github.com/user-attachments/assets/dfe1eac1-312c-434f-8a3f-92ff1579e742" width="300" height="200" />
</p>


**Temper Detectore:**
1. Select a suspected image (`.png` or `.tiff`).
2. Select the metadata file
    - If the image is tempered the program return Fasle along with the number of mismatches, Inliers value [0,1], and the path it has stored the overlay image.
    - Otherwise will return True and the number of mismatches, Inliers value [0,1], and the path it has stored the overlay image.
3. The program will generate:
   - A overlay image which has red circle around mismatch points: `res/<name>_overlay.png`
<p float="left">
  <img src="https://github.com/user-attachments/assets/b6ae53b7-9df2-4ceb-b24e-4d67fc8ec86a" width="300" height="200" />
  <img src="https://github.com/user-attachments/assets/6a87e155-0256-495f-aab4-90c40c483c3a" width="300" height="200" />
</p>


