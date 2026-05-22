#!/usr/bin/env python3
"""
Open-Vocabulary 3D Object Localization.

Grounding DINO (text → 2D box) + FFS stereo depth (box → distance).

Type what you want to find. The system draws boxes with real-time distance.

Usage:
    python scripts/open_vocab_3d.py --text "person . chair . monitor . keyboard"
    python scripts/open_vocab_3d.py --text "cup . phone . laptop"

Requirements:
    - groundingdino_swint_ogc.pth in weights/
    - bert-base-uncased/ in project root (config.json, model.safetensors, vocab.txt, tokenizer_config.json)
    - transformers==4.36.0 (downgraded for compatibility)
"""

import os, sys, cv2, numpy as np, argparse, time, pickle, onnxruntime as ort
import torch

# ---- Patch GroundingDINO to use local BERT ----
import transformers.models.auto.tokenization_auto as _tauto
_orig_tok = _tauto.AutoTokenizer.from_pretrained

def _patched_tok(name, *a, **kw):
    if name == "bert-base-uncased":
        name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bert-base-uncased")
    return _orig_tok(name, *a, **kw)

_tauto.AutoTokenizer.from_pretrained = _patched_tok
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from groundingdino.util.inference import Model as GDModel

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
COLORS = [
    (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
]


def load_gd_model():
    import groundingdino.config
    config_path = os.path.join(os.path.dirname(groundingdino.config.__file__),
                               "GroundingDINO_SwinT_OGC.py")
    weight_path = os.path.join(code_dir, "weights", "groundingdino_swint_ogc.pth")
    return GDModel(model_config_path=config_path, model_checkpoint_path=weight_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--images", type=str, default="", help="Use image directory instead of video")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--ffs_model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="320x736")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--zfar", type=float, default=6)
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--text", type=str, default="person . chair . table . monitor . keyboard")
    parser.add_argument("--gd_every", type=int, default=15, help="Run GD every N frames")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save", type=str, default="", help="Save output video (e.g. output/demo.mp4)")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    queries = [t.strip() for t in args.text.split(".")]

    # ---- FFS TRT ----
    mh, mw = map(int, args.res.split("x"))
    onnx = f"weights/onnx/{args.ffs_model}/{args.res}/{args.ffs_model}_iters_{args.iters}_res_{args.res}.onnx"
    session = ort.InferenceSession(onnx, ort.SessionOptions(), providers=[
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ])
    ds = args.half_w / float(mw)
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline, fx = calib["baseline"], calib["K_left"][0, 0]

    # ---- Grounding DINO ----
    print("Loading Grounding DINO...")
    gd = load_gd_model()
    print(f"Queries: {queries}")

    # ---- Video or Images ----
    image_paths = []
    if args.images:
        image_paths = sorted([os.path.join(args.images, f) for f in os.listdir(args.images)
                              if f.endswith(('.png','.jpg','.jpeg','.bmp'))])
        n_frames = len(image_paths)
        cap = None
        print(f"Images: {n_frames} frames from {args.images}")
    else:
        cap = cv2.VideoCapture(args.video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {n_frames} frames")

    # ---- Buffers ----
    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)
    def norm(img, buf):
        np.multiply(img, 1.0/255.0, out=buf)
        np.subtract(buf, MEAN, out=buf); np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2, 0, 1)[None])

    # Warmup
    dw = norm(np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8), buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    # ---- State ----
    frame_idx, times = 0, []
    disp_smooth = None
    disp_lo, disp_hi = None, None
    depth_max = None
    prev_gray = None
    tracked_boxes = []

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    if not args.no_display:
        cv2.namedWindow("Open-Vocab 3D Localization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Open-Vocab 3D Localization", 1920, 1080)
    print("Running... q=quit", flush=True)

    while True:
        if args.images:
            if frame_idx >= len(image_paths):
                frame_idx = 0  # loop
            frame = cv2.imread(image_paths[frame_idx])
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break
        t0 = time.perf_counter()

        left_raw = frame[:, :args.half_w]
        right_raw = frame[:, args.half_w:]

        # ---- FFS Depth ----
        lr = cv2.resize(left_raw, (mw, mh), interpolation=cv2.INTER_LINEAR)
        rr = cv2.resize(right_raw, (mw, mh), interpolation=cv2.INTER_LINEAR)
        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]
        if disp_smooth is not None:
            disp_smooth = args.temporal * disp + (1 - args.temporal) * disp_smooth
        else:
            disp_smooth = disp.copy()
        disp_full = cv2.resize(disp_smooth, (args.half_w, left_raw.shape[0]),
                               interpolation=cv2.INTER_LINEAR) * ds
        depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        # ---- Grounding DINO (periodic) + optical flow tracking ----
        gray = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
        if frame_idx % args.gd_every == 0:
            detections = gd.predict_with_classes(
                image=left_raw, classes=queries,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            tracked_boxes = []
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i, box in enumerate(detections.xyxy):
                    if hasattr(detections, 'class_id') and detections.class_id[i] is not None:
                        cls_id = int(detections.class_id[i])
                    else:
                        cls_id = 0
                    if cls_id < len(queries):
                        x1,y1,x2,y2 = [int(b) for b in box]
                        tracked_boxes.append({
                            'box': [x1,y1,x2,y2], 'qid': cls_id,
                            'pts': np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]),
                        })
        elif prev_gray is not None and tracked_boxes:
            for tb in tracked_boxes:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, tb['pts'], None,
                    winSize=(31, 31), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                if (status == 1).sum() >= 3:
                    tb['pts'] = new_pts
                    tb['box'] = [int(new_pts[:,0].min()), int(new_pts[:,1].min()),
                                 int(new_pts[:,0].max()), int(new_pts[:,1].max())]

        # ---- Draw ----
        vis = left_raw.copy()
        annotated = []
        for tb in tracked_boxes:
            x1, y1, x2, y2 = tb['box']
            bw, bh = x2 - x1, y2 - y1
            # Sample depth from inner 40% of box (avoid including background)
            margin_x = int(bw * 0.3)
            margin_y = int(bh * 0.3)
            cx1 = max(0, x1 + margin_x)
            cx2 = max(cx1 + 1, min(x2 - margin_x, depth.shape[1] - 1))
            cy1 = max(0, y1 + margin_y)
            cy2 = max(cy1 + 1, min(y2 - margin_y, depth.shape[0] - 1))
            roi = depth[cy1:cy2, cx1:cx2]
            vd = roi[(roi > 0.2) & (roi < args.zfar)]
            if len(vd) < 5:
                # Fallback: use center 20% of box
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                rw, rh = max(1, bw//5), max(1, bh//5)
                cx1 = max(0, cx - rw); cx2 = min(depth.shape[1]-1, cx + rw)
                cy1 = max(0, cy - rh); cy2 = min(depth.shape[0]-1, cy + rh)
                roi = depth[cy1:cy2, cx1:cx2]
                vd = roi[(roi > 0.2) & (roi < args.zfar)]
            if len(vd) < 3:
                continue
            # Use 30th percentile (closer to object surface, less background)
            d = np.percentile(vd, 30)

            qid = tb['qid']
            label = f"{queries[qid]} {d:.2f}m"
            color = COLORS[qid % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            annotated.append({'name': queries[qid], 'distance': d})

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Panel (cached colormap ranges) ----
        v = disp_full > 0.5
        if v.any():
            if disp_lo is None:
                disp_lo = np.percentile(disp_full[v], 2)
                disp_hi = np.percentile(disp_full[v], 98)
        disp_vis = vis_disparity(disp_full, min_val=disp_lo or 0, max_val=disp_hi or 1)

        vd = depth > 0
        if vd.any() and depth_max is None:
            depth_max = min(args.zfar, np.percentile(depth[vd], 95))
        depth_vis = vis_disparity(depth, min_val=0, max_val=depth_max or args.zfar,
                                   color_map=cv2.COLORMAP_JET)

        panel = np.hstack([vis, disp_vis, depth_vis])
        avg_ms = np.mean(times[-15:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        cv2.putText(panel, f'Find: {args.text} | {fps:.0f}FPS | {len(annotated)} found',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        for i, a in enumerate(annotated[:8]):
            cv2.putText(panel, f"  {a['name']}: {a['distance']:.2f}m",
                        (10, 52+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save,
                    cv2.VideoWriter_fourcc(*'XVID'), 15,
                    (panel.shape[1], panel.shape[0]))
            writer.write(panel)

        if not args.no_display:
            cv2.imshow("Open-Vocab 3D Localization", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        prev_gray = gray.copy()
        frame_idx += 1

    if cap is not None: cap.release()
    if writer:
        writer.release()
        print(f"Video saved: {args.save}")
    if not args.no_display:
        cv2.destroyAllWindows()
    avg_ms = np.mean(times[3:]) * 1000 if len(times) > 3 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
