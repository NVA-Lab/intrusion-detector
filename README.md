# Real-Time Intrusion Detection Pipeline

This repository implements a four-stage end-to-end system for detecting and responding to unauthorized entry:

1. **CSI-Based Presence Detection**  
   Continuously monitor WiFi CSI (Channel State Information) to detect door openings or human entry.

2. **Intruder Localization**  
   Analyze phase and amplitude shifts to approximate the intruder’s position in the room.

3. **Real-Time Human Detection & Tracking**  
   - **YOLOv8** for ultra-fast person bounding-box detection  
   - **SAM2 + SAMURAI** for pixel-precise segmentation, centroid extraction, and PTZ camera control
4.  **Video Recording & Behavioral Analysis**  
   - **Automated Recording**: Trigger 5-second video clips when suspicious activity is detected  
   - **DAM (Dynamic Action Model)**: Generate natural language descriptions of human actions and behaviors from recorded video segments  
5. **Logging & Anomaly Alerts**  
   Record intrusion events, track “stationary interaction” behaviors, and push logs/alerts to the dashboard or mobile.

By combining wireless sensing, computer vision, and intelligent logging, this pipeline delivers robust, automated intrusion monitoring in real time.  

### About SAM2
**SAM2** (Segment Anything Model 2) is designed for object segmentation and tracking but lacks built-in capabilities 
for performing this in real time.

### About SAMURAI
**SAMURAI** enhances SAM2 by introducing motion modeling, leveraging temporal motion cues for better 
tracking accuracy without retraining or fine-tuning.  


## Key Features

- **YOLOv8 Person Detection**  
  Ultra-fast bounding-box detection of humans in each video frame.

- **SAM2 Segmentation & Tracking**  
  Pixel-accurate masks + centroid extraction to hand off to the PTZ controller.

- **Motion-Aware Tracking**  
  SAMURAI motion modeling ensures stable multi-object tracks without retraining.

- **Anomaly Alerting**  
  Detects “stop-and-interact” behavior (e.g., a thief grabbing an object) and generates an alert.


---

## Setup Instructions
I recommend using uv venv to create isolated environments, simplifying dependency management and ensuring reproducible setups.

### 1. Create & activate virtualenv
```bash
# Install the 'uv' CLI and create a new venv
pip install uv
uv venv

# On macOS / Linux
source .venv/bin/activate
# On Windows (PowerShell)
source .venv/Scripts/activate
```

### 2. Clone the repository
```
git clone https://github.com/NVA-Lab/intrusion-detector.git
```

### 3. Install packages
```
cd intrusion-detector

# Install the core package (SAM2 + demo app) in editable mode
uv pip install -e .


### 4. Download SAM2 Checkpoints
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

---

### Acknowledgment
This project leverages:  
- **YOLOv8** by Ultralytics for ultra-fast real-time person detection.  
- **SAM2** by Meta FAIR for pixel-precise segmentation and tracking.  
- **SAMURAI** by the University of Washington’s Information Processing Lab for motion-aware memory modeling.  
- **DAM (Describe Anything Model)** by NVIDIA for concise natural language descriptions of subject actions or state changes in videos.

## Citation
```
@article{glenn2024yolov8,
  title={YOLOv8: Next-Generation Real-Time Object Detection},
  author={Glenn Jocher and Ultralytics},
  year={2024},
  url={https://github.com/ultralytics/ultralytics}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi et al.},
  year={2024},
  url={https://arxiv.org/abs/2408.00714}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting SAM for Zero-Shot Visual Tracking with Motion-Aware Memory},
  author={Yang et al.},
  year={2024},
  url={https://arxiv.org/abs/2411.11922}
}

```

---
