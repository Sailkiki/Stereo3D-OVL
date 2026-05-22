#!/usr/bin/env python3
"""
Real-time single-frame point cloud from stereo video.

Each frame produces a clean standalone point cloud. No multi-frame
fusion = no drift, no duplicates, no mess.

Usage:
    python scripts/realtime_recon.py --model 20_26_39 --res 320x736 --save_pc
"""

import os, sys, cv2, numpy as np, argparse, pickle, time, onnxruntime as ort
import open3d as o3d

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
SCALE = np.float32(1.0 / 255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="320x736")
    parser.add_argument("--iters", type=int, default=4, choices=[4, 8])
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--zfar", type=float, default=5, help="max depth (m)")
    parser.add_argument("--zmin", type=float, default=0.2, help="min depth (m)")
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--pc_step", type=int, default=4, help="downsample: 1=full, 4=faster")
    parser.add_argument("--save_pc", action="store_true", help="save all point clouds as PLY")
    parser.add_argument("--save_dir", type=str, default="output/pc")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- Calibration ----
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline = calib["baseline"]
    P1 = calib["P1"]
    fx, fy, cx, cy = P1[0,0], P1[1,1], P1[0,2], P1[1,2]
    maps = np.load(os.path.join(os.path.dirname(args.calib), "rectification_maps.npz"))
    map_lx, map_ly = maps["map_lx"], maps["map_ly"]
    map_rx, map_ry = maps["map_rx"], maps["map_ry"]

    # ---- TRT Model ----
    onnx_path = f"weights/onnx/{args.model}/{args.res}/{args.model}_iters_{args.iters}_res_{args.res}.onnx"
    providers = [
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ]
    session = ort.InferenceSession(onnx_path, ort.SessionOptions(), providers=providers)
    _, _, mh, mw = session.get_inputs()[0].shape
    disp_scale = args.half_w / float(mw)
    print(f"Model: {mh}x{mw} | Baseline={baseline*1000:.0f}mm | fx={fx:.0f} fy={fy:.0f}")

    # ---- Video ----
    cap = cv2.VideoCapture(args.video)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames: {n_frames} | Step: {args.pc_step} | z=[{args.zmin},{args.zfar}]m")

    # ---- Buffers ----
    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)

    def norm(img, buf):
        np.multiply(img, SCALE, out=buf)
        np.subtract(buf, MEAN, out=buf)
        np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2, 0, 1)[None])

    # Warmup
    d = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
    dw = norm(d, buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    # ---- K for point cloud back-projection (scaled to step) ----
    K_s = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    K_s[:2] /= args.pc_step

    # ---- State ----
    frame_idx = 0
    times = []
    disp_smooth = None
    disp_lo, disp_hi = None, None

    print("q=quit  s=save current PC  space=screenshot", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()

        # Rectify
        left = cv2.remap(frame[:, :args.half_w], map_lx, map_ly, cv2.INTER_LINEAR)
        right = cv2.remap(frame[:, args.half_w:], map_rx, map_ry, cv2.INTER_LINEAR)

        # Resize + normalize
        lr = cv2.resize(left, (mw, mh), interpolation=cv2.INTER_LINEAR)
        rr = cv2.resize(right, (mw, mh), interpolation=cv2.INTER_LINEAR)
        ln = norm(lr, buf_l); rn = norm(rr, buf_r)

        # TRT inference
        out = session.run(['disparity'], {'left_image': ln, 'right_image': rn})
        disp = out[0][0, 0]

        # Smooth + upsample
        if disp_smooth is not None:
            disp_smooth = args.temporal * disp + (1 - args.temporal) * disp_smooth
        else:
            disp_smooth = disp.copy()
        disp_full = cv2.resize(disp_smooth, (args.half_w, left.shape[0]),
                               interpolation=cv2.INTER_LINEAR) * disp_scale
        depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Point Cloud (single frame, no accumulation) ----
        depth_pc = depth[::args.pc_step, ::args.pc_step]
        color_pc = cv2.cvtColor(left[::args.pc_step, ::args.pc_step], cv2.COLOR_BGR2RGB)
        pts = depth2xyzmap(depth_pc, K_s).reshape(-1, 3)
        valid = (pts[:, 2] > args.zmin) & (pts[:, 2] < args.zfar)
        pcd = toOpen3dCloud(pts[valid], color_pc.reshape(-1, 3)[valid])

        if args.save_pc:
            o3d.io.write_point_cloud(f"{args.save_dir}/frame_{frame_idx:06d}.ply", pcd)

        # ---- Display ----
        valid_d = disp_full > 0.5
        if valid_d.any():
            if disp_lo is None:
                disp_lo, disp_hi = np.percentile(disp_full[valid_d], 2), np.percentile(disp_full[valid_d], 98)
        disp_vis = vis_disparity(disp_full, min_val=disp_lo or 0, max_val=disp_hi or 1)
        top = np.hstack([left, right, disp_vis])
        avg_ms = np.mean(times[-30:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        vd = depth > 0
        d_mean = depth[vd].mean() if vd.any() else 0
        n_pts = len(pts[valid])
        info = f"FPS:{fps:.0f} | depth:{d_mean:.2f}m | pts:{n_pts} | frame:{frame_idx}"
        cv2.putText(top, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if not args.no_display:
            s = 1024 / top.shape[1]
            cv2.imshow("Depth + Point Cloud", cv2.resize(top,
                        (int(top.shape[1]*s), int(top.shape[0]*s))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord("s"):
                path = f"{args.save_dir}/frame_{frame_idx:06d}.ply"
                o3d.io.write_point_cloud(path, pcd)
                print(f"  Saved {path} ({n_pts} pts)", flush=True)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: {avg_ms:.0f}ms | {n_pts} pts | depth={d_mean:.2f}m", flush=True)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    avg_ms = np.mean(times[5:]) * 1000 if len(times) > 5 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
