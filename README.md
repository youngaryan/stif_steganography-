# EyeDetector—COM31006 Watermarking Project

**Author:** Aryan Golbaghi  
**Module:** COM31006 — Image-to-Image Steganography for Watermark Creation
**Purpose:** This project demonstrates image watermark embedding, verification, tamper detection, and recovery using topn N non-overlapping SIFT keypoints and altering the least significant bit (LSBs) steganography.

---

## Overview

EyeDetector allows users to:
- Embed a watermark into a carrier image.
- Verify the authenticity of a suspected image.
- Detect and visualize mismatched or tampered watermark regions.
- Recover the original watermark using majority voting.

The watermark is embedded in the **(LSBs)** of a **random channel** at **non-overlapping SIFT keypoints**, providing watermarking embedding, verification and temper detection.

---

## Features

- **Watermark Embedding** at strongest SIFT keypoints.
- **Watermark Verification** using geometric homography and bit-pattern similarity.
- **Tamper Detection** overlays the tempered points with red circle and generate mismatch analysis.
- **Watermark Recovery** recovers the water mark using majority technique voting.
- **Segment Size** allows user to choose segement size for embedding.
- **Simple GUI** with image preview and progress feedback.

---

## How It Works

### Embedding Process
- Convert watermark image to 9x9 binary.
- Detect strong SIFT keypoints in carrier image.
- For each selected keypoint:
  - Embed the watermark in a square patch (centered on the keypoint) into the **LSB of reandom channels**.
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
- `Segement Size Chooser`: Allows user to choose segment size.

The GUI also includes progress feedback and real-time image previews.

---

## Requirements
Before running the project, it's recommended to set up a virtual environment to avoid dependency conflicts:

1. **Create the environment** (e.g., `cv_env`):
   ```bash
   python -m venv cv_env
   ```
2. **Activate it:**

   - On **Windows**:
     ```bash
     cv_env\Scripts\activate
     ```

   - On **macOS/Linux**:
     ```bash
     source cv_env/bin/activate
     ```



3. **Install dependencies (opencv-python, numpy, Pillow):**
```bash
pip install -r requirements.txt
```
4. **Run the project:**
```bash
 python eye.py
```
---
## Project File Structure

```bash
stif_steganography-/
├── script/
│   ├── detecor.py
│   ├── embed.py
│   ├── helper.py
│   ├── ui.py
│   ├── verify.py
├── images/  #sample carrier and watermark image
│   ├── che.png
│   ├── watermark.png
├── res/  #folder which the genrated data(images and metadata) will be stored at
├── eye.py  #brain of the program
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

![image](https://github.com/user-attachments/assets/2e16ca4f-3f1e-42ee-aafa-d53750b687d3)


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
  <img src="https://github.com/user-attachments/assets/7666486e-5c74-4c66-8536-60f4d6950c83" width="700" height="400" />
  <img src="https://github.com/user-attachments/assets/f245e957-c1ed-482f-a551-58c9f5d1b051" width="700" height="400" />
</p>

**Temper Detectore:**
1. Select a suspected image (`.png` or `.tiff`).
2. Select the metadata file
    - If the image is tempered the program return Fasle along with the number of mismatches, Inliers value [0,1], and the path it has stored the overlay image.
    - Otherwise will return True and the number of mismatches, Inliers value [0,1], and the path it has stored the overlay image.
3. The program will generate:
   - A overlay image which has red circle around mismatch points: `res/<name>_overlay.png`
<p float="left">
  <img src="https://github.com/user-attachments/assets/a9568c87-6cae-4305-8cba-e8aea381a7e1" width="700" height="500" />
  <img src="https://github.com/user-attachments/assets/1d54cf49-8523-4965-8fe2-f217b836d80b" width="700" height="500" />
</p>

**Recover Watermark Image:**
1. Select a suspected image (`.png` or `.tiff`).
2. Select the metadata file
    - If the image is verified, the program will return the path of the recovered image.
    - Otherwise will raise an error.
<p float="left">
  <img src="https://github.com/user-attachments/assets/86e38e8c-6f30-4aef-ac04-adb0bd21420a" width="400" height="200" />
  <img src="https://github.com/user-attachments/assets/e19f1c8b-9249-4e42-8bcb-a98e3a700ee7" width="400" height="200" />
</p>

