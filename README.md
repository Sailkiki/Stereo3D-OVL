# Language-Driven Real-Time 3D Perception from Stereo Vision

[![Demo](https://img.shields.io/badge/Demo-Video-blue)](assets/office_demo.mp4)
[![KITTI](https://img.shields.io/badge/KITTI-Eval-green)](assets/kitti_demo.mp4)

An end-to-end real-time stereo depth perception system built on Fast-FoundationStereo (CVPR 2026). 

**Demo 1**:

<video src="assets/office_demo.mp4" controls width="100%"></video>

**Demo 2**:

<video src="assets/kitti_demo.mp4" controls width="100%"></video>

---

## What's Implemented

- Zhang's stereo calibration: chessboard corner detection, monocular/stereo calibration, rectification map generation.
- TensorRT real-time inference: ONNX Runtime + TensorRT FP16, 19ms inference at 320×736, 18-25 FPS end-to-end.
- Language-driven 3D localization: Grounding DINO + FFS depth + LK optical flow tracking.
- Supports stereo camera, video file, image sequence, and KITTI dataset input modes.

---

## Quick Start

### Environment

```bash
conda create -n ffs python=3.12 && conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com
pip install ultralytics groundingdino-py
```

### Download Weights

FFS weights from [NVIDIA Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap):
- `model_best_bp2_serialize.pth` → `weights/`
- `onnx/` → `weights/onnx/`

Grounding DINO from [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) → `weights/`.

BERT from [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased) → `bert-base-uncased/`.

### Calibration

```bash
python calibration/stereo_calibrate.py \
    --calib_dir calibration/ \
    --pattern_size "8,4" --square_size 0.013 \
    --output_dir calibration/calib_results/
```

### Real-Time Depth

```bash
python scripts/realtime_depth_trt.py --video your_video.avi --model 20_26_39 --res 320x736
```

### Language-Driven 3D Localization

```bash
# KITTI
python scripts/kitti_open_vocab.py --text "car . pedestrian . cyclist . truck"

# Office
python scripts/open_vocab_3d.py --video video/office.avi.avi --text "person . chair . monitor"
```

### KITTI Evaluation

```bash
python scripts/kitti_eval.py --kitti_dir path/to/KITTI --model 23_36_37 --iters 8
```

## Citation

**Fast-FoundationStereo**

Bowen Wen, Shaurya Dewan, Stan Birchfield.  
*Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching.* CVPR 2026.

- Paper: <https://arxiv.org/abs/2512.11130>
- Code: <https://github.com/NVlabs/Fast-FoundationStereo>

**Grounding DINO**

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang.  
*Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* ECCV 2024.

- Paper: <https://arxiv.org/abs/2303.05499>
- Code: <https://github.com/IDEA-Research/GroundingDINO>
