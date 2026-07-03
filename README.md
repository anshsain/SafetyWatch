# 🦺 SafetyWatch — Construction PPE Detection
 
Real-time helmet compliance detection using YOLOv8 with two-phase transfer learning. Detects whether workers on construction sites are wearing helmets, and reports a site-level compliance percentage.
 
**Live demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(update with your actual link)*
 
---
 
## Overview
 
| | |
|---|---|
| **Task** | Object detection (2 classes) |
| **Architecture** | YOLOv8n, two-phase transfer learning |
| **Dataset** | SHEL5K (Mendeley, DOI 10.17632/9rcv8mm682.4) |
| **mAP@50** | 89.6% |
| **Helmet AP@50** | 87.2% |
| **No-helmet AP@50** | 92.0% |
| **Deployment** | Streamlit Community Cloud |
 
---
 
## Classes
 
| Class | Meaning | Box style |
|---|---|---|
| `helmet` | Worker wearing a hard hat — compliant | Full head-region box |
| `no_helmet` | Bare head — safety violation | Full head-region box |
 
Both classes use the same full-head-region bounding box annotation style. This is deliberate — see the lessons-learned section below for why it matters.
 
---
 
## Results
 
| Metric | Value |
|---|---|
| mAP@50 | 89.6% |
| mAP@50-95 | 64.3% |
| Precision | 87.2% |
| Recall | 85.7% |
| Inference speed | ~4ms/image |
 
| Class | AP@50 | Test instances |
|---|---|---|
| helmet | 87.2% | 5,783 |
| no_helmet | 92.0% | 3,156 |
 
`no_helmet` AP is slightly higher than `helmet` despite being the minority class, suggesting bare heads are a visually distinctive pattern that the model converges on efficiently even with less data.
 
---
 
## Architecture & Training Strategy
 
**Two-phase transfer learning** on YOLOv8n pretrained on COCO:
 
**Phase 1 (15 epochs):** First 9 backbone layers frozen. Only the neck and detection head train, stabilising the new task-specific layers before touching the pretrained backbone features. LR: 1e-3.
 
**Phase 2 (50 epochs):** All layers unfrozen. Full end-to-end fine-tuning at 10× lower LR (1e-4), allowing the backbone to adapt to the construction-site domain without catastrophic forgetting.
 
---
 
## Lessons Learned — Why the Dataset Choice Matters
 
The first version of this model was trained on the **Hard Hat Workers dataset** (Roboflow, 7,000+ images). Post-deployment testing revealed a systematic failure: the model correctly identified helmeted workers as violations on real-world images with near-perfect confidence.
 
**Root cause — annotation convention mismatch.** In that dataset, `helmet` was annotated as a tight box around just the helmet piece (small, top-of-head), while `head` was annotated as a larger box covering the entire face and head region. The model learned this **box-size convention** rather than the actual safety concept — when it saw a head-region box at inference time, it predicted `head` (violation) regardless of whether a helmet was visible, because that's what matched the training distribution.
 
This failure was invisible in offline evaluation because the test set came from the same dataset with the same annotation style, so IoU matching worked correctly and yielded a plausible-looking 64.3% mAP. But it failed entirely on independently sourced construction-site photos.
 
**The fix — SHEL5K.** SHEL5K (Safety Helmet and ELD-5K) explicitly separates `head with helmet` and `head without helmet` as distinct classes using consistent full-head-region boxes throughout — forcing the model to learn whether a helmet is present on a head, not the shape of the box drawn around it. Retraining on SHEL5K raised mAP to 89.6% and, more importantly, produced a model that generalises correctly to real-world images across all helmet colours and angles.
 
**Key takeaway.** Offline metrics don't always predict production behaviour. Distribution shift caused by annotation-convention inconsistency between the training dataset and inference-time inputs is a real failure mode that doesn't surface in held-out test sets drawn from the same annotation pool. The correct diagnostic was testing on independently sourced images, not on the original dataset's test split.
 
---
 
## Project Structure
 
```
.
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── best.pt                     # Trained YOLOv8n weights (SHEL5K, two-phase)
└── notebooks/
    ├── layer1_setup.ipynb         # SHEL5K download, annotation remapping, EDA
    ├── layer2_baseline.ipynb      # Baseline YOLOv8n training + evaluation
    ├── layer3_resnet50.ipynb      # Two-phase transfer learning
    ├── layer4_gradcam.ipynb       # GRAD-CAM, PR curves, confusion matrix
    └── layer5_deploy.ipynb        # Streamlit deployment automation
```
 
---
 
## Running Locally
 
```bash
git clone https://github.com/your-username/safetywatch-ppe.git
cd safetywatch-ppe
pip install -r requirements.txt
streamlit run app.py
```
 
---
 
## App Features
 
**Image detection.** Upload a JPG/PNG → bounding boxes with class labels → compliance dashboard (helmet count, violation count, % compliance, inference time in ms).
 
**Video detection.** Upload MP4/AVI/MOV → frame-by-frame detection → downloadable annotated video.
 
**Model insights.** Per-class AP bars, overall metrics table, training strategy summary.
 
---
 
## Tech Stack
 
PyTorch · Ultralytics YOLOv8 · OpenCV · Streamlit · SHEL5K · Google Colab (T4 GPU)
