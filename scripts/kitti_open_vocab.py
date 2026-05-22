#!/usr/bin/env python3
"""
Open-Vocabulary 3D Object Localization on KITTI.

Grounding DINO detects objects by text → FFS gives distance.

Usage:
    python scripts/kitti_open_vocab.py --text "car . pedestrian . cyclist . building . tree"
    python scripts/kitti_open_vocab.py --text "truck . van . traffic light . road sign"
"""

import os, sys, cv2, numpy as np, argparse, time, pickle, onnxruntime as ort

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

FX, FY, CX, CY = 721.5377, 721.5377, 609.5593, 172.854
BASELINE = 0.535164
MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
COLORS = [
    (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (255, 128, 128), (128, 255, 255),
]


def letterbox(img, tw, th):
    h, w = img.shape[:2]
    s = min(tw/w, th/h); nw, nh = int(w*s), int(h*s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    oy, ox = (th-nh)//2, (tw-nw)//2
    canvas[oy:oy+nh, ox:ox+nw] = r
    return canvas, s, ox, oy


def load_gd_model():
    import groundingdino.config
    c = os.path.join(os.path.dirname(groundingdino.config.__file__), "GroundingDINO_SwinT_OGC.py")
    w = os.path.join(code_dir, "weights", "groundingdino_swint_ogc.pth")
    return GDModel(model_config_path=c, model_checkpoint_path=w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="../libSGM/data/2011_09_26_drive_0001_sync")
    parser.add_argument("--ffs_model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="320x736")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--zfar", type=float, default=80)
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--text", type=str, default="car . pedestrian . cyclist . truck . traffic light")
    parser.add_argument("--gd_every", type=int, default=15, help="Run GD every N frames (higher=faster)")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save", type=str, default="", help="Save output video to path (e.g. output/demo.mp4)")
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

    # ---- Grounding DINO ----
    print("Loading Grounding DINO...")
    gd = load_gd_model()
    print(f"Queries: {queries}")

    # ---- KITTI images ----
    left_dir = os.path.join(args.kitti_dir, "image_02/data")
    right_dir = os.path.join(args.kitti_dir, "image_03/data")
    files = sorted(os.listdir(left_dir))
    n_frames = len(files)
    print(f"KITTI: {n_frames} frames | GD every {args.gd_every}")

    # ---- Buffers ----
    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)
    def norm(img, buf):
        np.multiply(img, 1.0/255.0, out=buf)
        np.subtract(buf, MEAN, out=buf); np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2,0,1)[None])

    dw = norm(np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8), buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    # ---- State ----
    frame_idx, times = 0, []
    disp_smooth = None
    disp_lo, disp_hi = None, None
    depth_max = None
    # Optical flow tracking state
    prev_gray = None
    tracked_boxes = []  # [(x1,y1,x2,y2,qid), ...] updated by flow

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    if not args.no_display:
        cv2.namedWindow("KITTI Open-Vocab 3D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("KITTI Open-Vocab 3D", 1920, 1080)
    print("Running... q=quit", flush=True)

    while True:
        loop_idx = frame_idx % n_frames
        if args.save and frame_idx >= n_frames:
            break  # no loop when saving video
        if loop_idx == 0 and frame_idx > 0:
            prev_gray = None; tracked_boxes = []; disp_smooth = None
            disp_lo = disp_hi = depth_max = None
        t0 = time.perf_counter()
        left = cv2.imread(os.path.join(left_dir, files[loop_idx]))
        right = cv2.imread(os.path.join(right_dir, files[loop_idx]))
        h_orig, w_orig = left.shape[:2]

        # Letterbox + normalize
        lr, s, ox, oy = letterbox(left, mw, mh)
        rr, _, _, _ = letterbox(right, mw, mh)
        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]
        nw, nh = int(w_orig * s), int(h_orig * s)
        disp = disp[oy:oy+nh, ox:ox+nw] * (w_orig / nw)
        disp_full = cv2.resize(disp, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        if disp_smooth is not None:
            disp_smooth = args.temporal * disp_full + (1-args.temporal) * disp_smooth
        else:
            disp_smooth = disp_full.copy()
        depth = np.where(disp_smooth > 0.5, FX * BASELINE / disp_smooth, 0)

        # ---- Grounding DINO (periodic) ----
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if frame_idx % args.gd_every == 0:
            detections = gd.predict_with_classes(
                image=left, classes=queries,
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

        # ---- Optical flow: track boxes between GD refreshes ----
        elif prev_gray is not None and tracked_boxes:
            for tb in tracked_boxes:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, tb['pts'], None,
                    winSize=(31, 31), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                if (status == 1).sum() >= 3:
                    tb['pts'] = new_pts
                    x1 = int(new_pts[:,0].min()); y1 = int(new_pts[:,1].min())
                    x2 = int(new_pts[:,0].max()); y2 = int(new_pts[:,1].max())
                    tb['box'] = [x1, y1, x2, y2]

        # ---- Draw boxes with distance ----
        vis = left.copy()
        annotated = []
        for tb in tracked_boxes:
            x1, y1, x2, y2 = tb['box']
            bw, bh = x2-x1, y2-y1
            # Sample depth from inner 40% of box
            mx, my = int(bw*0.3), int(bh*0.3)
            ix1 = max(0, x1+mx); iy1 = max(0, y1+my)
            ix2 = max(ix1+1, min(x2-mx, depth.shape[1]-1))
            iy2 = max(iy1+1, min(y2-my, depth.shape[0]-1))
            roi = depth[iy1:iy2, ix1:ix2]
            vd = roi[(roi > 0.5) & (roi < args.zfar)]
            if len(vd) < 5:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                rw, rh = max(1, bw//5), max(1, bh//5)
                ix1 = max(0, cx-rw); ix2 = min(depth.shape[1]-1, cx+rw)
                iy1 = max(0, cy-rh); iy2 = min(depth.shape[0]-1, cy+rh)
                vd = depth[iy1:iy2, ix1:ix2]
                vd = vd[(vd > 0.5) & (vd < args.zfar)]
            if len(vd) < 3:
                continue
            d = np.percentile(vd, 30)

            qid = tb['qid']
            label = f"{queries[qid]} {d:.1f}m"
            color = COLORS[qid % len(COLORS)]
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1,y1-th-6), (x1+tw+4,y1), color, -1)
            cv2.putText(vis, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            annotated.append({'name': queries[qid], 'distance': d})

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Display (ranges computed once then cached) ----
        v = disp_smooth > 0.5
        if v.any():
            if disp_lo is None:
                disp_lo = np.percentile(disp_smooth[v], 2)
                disp_hi = np.percentile(disp_smooth[v], 98)
        disp_vis = vis_disparity(disp_smooth, min_val=disp_lo or 0, max_val=disp_hi or 1)

        vd = depth > 0
        if vd.any() and depth_max is None:
            depth_max = min(args.zfar, np.percentile(depth[vd], 95))
        depth_vis = vis_disparity(depth, min_val=0, max_val=depth_max or args.zfar,
                                   color_map=cv2.COLORMAP_JET)
        panel = np.hstack([vis, disp_vis, depth_vis])
        avg_ms = np.mean(times[-15:])*1000 if times else 0
        fps = 1000.0/avg_ms if avg_ms>0 else 0
        cv2.putText(panel, f'KITTI: {args.text} | {fps:.0f}FPS | {len(annotated)} detections',
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        for i, a in enumerate(annotated[:10]):
            cv2.putText(panel, f"  {a['name']}: {a['distance']:.1f}m",
                        (10, 48+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        if not args.no_display:
            cv2.imshow("KITTI Open-Vocab 3D", panel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save,
                    cv2.VideoWriter_fourcc(*'XVID'), 15,
                    (panel.shape[1], panel.shape[0]))
            writer.write(panel)

        if args.no_display and frame_idx % 30 == 0:
            avg_ms = np.mean(times[-30:]) * 1000 if times else 0
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            print(f"  Frame {frame_idx}: {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS) | {len(annotated)} detected", flush=True)

        prev_gray = gray.copy()
        frame_idx += 1

    if writer:
        writer.release()
        print(f"Video saved: {args.save}")
    if not args.no_display:
        cv2.destroyAllWindows()
    avg_ms = np.mean(times[3:])*1000 if len(times)>3 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
