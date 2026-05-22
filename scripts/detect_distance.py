#!/usr/bin/env python3
"""
YOLO 2D detection + FFS depth → obstacle distance on your own video.

For each YOLO detection, sample FFS depth at the bounding box center
and display the distance. Clean, real-time, no clustering mess.

Usage:
    python scripts/detect_distance.py --video video/office.avi.avi --model 20_26_39 --res 320x736
"""

import os, sys, cv2, numpy as np, argparse, time, onnxruntime as ort
from ultralytics import YOLO

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="320x736")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--zfar", type=float, default=10)
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--yolo_model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- Calibration ----
    import pickle
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline = calib["baseline"]
    fx = calib["K_left"][0, 0]

    # ---- TRT ----
    mh, mw = map(int, args.res.split("x"))
    onnx = f"weights/onnx/{args.model}/{args.res}/{args.model}_iters_{args.iters}_res_{args.res}.onnx"
    session = ort.InferenceSession(onnx, ort.SessionOptions(), providers=[
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ])
    disp_scale = args.half_w / float(mw)
    print(f"FFS: {mh}x{mw} TRT | fx={fx:.0f} baseline={baseline*1000:.0f}mm")

    # ---- YOLO ----
    yolo = YOLO(args.yolo_model)
    print(f"YOLO: {args.yolo_model} conf={args.conf}")

    # ---- Video ----
    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {n_frames} frames @ {vid_fps:.1f} fps")

    # ---- Buffers ----
    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)
    def norm(img, buf):
        np.multiply(img, 1.0/255.0, out=buf)
        np.subtract(buf, MEAN, out=buf)
        np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2, 0, 1)[None])

    # Warmup
    d = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
    dw = norm(d, buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    # ---- State ----
    frame_idx = 0
    times = []
    disp_smooth = None
    disp_lo, disp_hi = None, None

    COLOR_MAP = {
        0: (0, 255, 0),     # person = green
        2: (255, 255, 0),   # car = cyan
        5: (255, 0, 255),   # bus = magenta
        7: (0, 255, 255),   # truck = yellow
    }

    cv2.namedWindow("Obstacle Detection + Distance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection + Distance", 1600, 500)
    print("Running... (q=quit t=smooth)", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()

        left_raw = frame[:, :args.half_w]
        right_raw = frame[:, args.half_w:]

        # Resize
        lr = cv2.resize(left_raw, (mw, mh), interpolation=cv2.INTER_LINEAR)
        rr = cv2.resize(right_raw, (mw, mh), interpolation=cv2.INTER_LINEAR)

        # FFS inference
        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]

        # Smooth + upsample
        if disp_smooth is not None:
            disp_smooth = args.temporal * disp + (1 - args.temporal) * disp_smooth
        else:
            disp_smooth = disp.copy()

        disp_full = cv2.resize(disp_smooth, (args.half_w, left_raw.shape[0]),
                               interpolation=cv2.INTER_LINEAR) * disp_scale

        # Depth at original left image resolution
        depth_full = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        # ---- YOLO detection on left image ----
        results = yolo(left_raw, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs):
                if conf < args.conf:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                # Sample depth at box center
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                if 0 <= cy_box < depth_full.shape[0] and 0 <= cx_box < depth_full.shape[1]:
                    d = depth_full[cy_box, cx_box]
                    if d > 0.1 and d < args.zfar:
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'distance': d,
                            'class': cls_id,
                            'name': results.names.get(cls_id, f'cls{cls_id}'),
                            'conf': conf,
                        })

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Visualization ----
        vis = left_raw.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = COLOR_MAP.get(det['class'], (0, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det['name']} {det['distance']:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Disparity visualization
        v = disp_full > 0.5
        if v.any():
            if disp_lo is None:
                disp_lo = np.percentile(disp_full[v], 2)
                disp_hi = np.percentile(disp_full[v], 98)
        disp_vis = vis_disparity(disp_full, min_val=disp_lo or 0, max_val=disp_hi or 1)

        # Depth colormap
        vd = depth_full > 0
        d_max = min(args.zfar, np.percentile(depth_full[vd], 95)) if vd.any() else args.zfar
        depth_vis = vis_disparity(depth_full, min_val=0, max_val=d_max, color_map=cv2.COLORMAP_JET)

        # Panel: Detection | Disparity | Depth
        panel = np.hstack([vis, disp_vis, depth_vis])
        avg_ms = np.mean(times[-15:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        info = f"{fps:.0f}FPS | {len(detections)} objects detected"
        cv2.putText(panel, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        for i, det in enumerate(detections[:8]):
            text = f"  {det['name']}: {det['distance']:.2f}m (conf={det['conf']:.2f})"
            cv2.putText(panel, text, (10, 45 + i * 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.imshow("Obstacle Detection + Distance", panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            if disp_smooth is not None:
                disp_smooth = None
            else:
                disp_smooth = disp.copy()

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    avg_ms = np.mean(times[3:]) * 1000 if len(times) > 3 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
