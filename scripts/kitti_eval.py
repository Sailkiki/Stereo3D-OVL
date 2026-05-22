"""
FFS stereo depth evaluation on KITTI with LiDAR ground truth.

KITTI images are pre-rectified, no calibration pipeline needed.
Uses known KITTI calibration (721.5px focal, 0.535m baseline).

Usage:
    python scripts/kitti_eval.py --kitti_dir ../libSGM/data/2011_09_26_drive_0001_sync
"""

import os, sys, cv2, numpy as np, argparse, time, onnxruntime as ort

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
SCALE = np.float32(1.0 / 255.0)


def load_kitti_lidar(bin_path):
    """Load KITTI LiDAR point cloud (.bin file), returns Nx4 (x,y,z,reflectance)."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def project_lidar_to_camera(lidar_pts, T_cam_velo):
    """Project LiDAR points (Nx3) to camera frame using T_cam_velo (3x4)."""
    n = len(lidar_pts)
    pts_h = np.hstack([lidar_pts[:, :3], np.ones((n, 1))])
    cam_pts = (T_cam_velo @ pts_h.T).T  # (N, 3)
    return cam_pts


def evaluate_depth(depth_est, depth_gt, mask):
    """Compute error metrics between estimated and GT depth."""
    d_est = depth_est[mask]
    d_gt = depth_gt[mask]
    if len(d_gt) < 100:
        return {}
    diff = np.abs(d_est - d_gt)
    rel = diff / d_gt

    metrics = {
        "RMSE": np.sqrt(np.mean(diff ** 2)),
        "MAE": np.mean(diff),
        "Rel": np.mean(rel),
        "a1": np.mean(np.maximum(d_est/d_gt, d_gt/d_est) < 1.25**1),
        "a2": np.mean(np.maximum(d_est/d_gt, d_gt/d_est) < 1.25**2),
        "a3": np.mean(np.maximum(d_est/d_gt, d_gt/d_est) < 1.25**3),
        "pixels": len(d_gt),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str,
                        default="../libSGM/data/2011_09_26_drive_0001_sync")
    parser.add_argument("--model", type=str, default="23_36_37")
    parser.add_argument("--res", type=str, default="576x960")
    parser.add_argument("--iters", type=int, default=8, help="8=best quality for evaluation")
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--zfar", type=float, default=80, help="max depth for KITTI (meters)")
    parser.add_argument("--max_frames", type=int, default=0, help="0=all frames")
    parser.add_argument("--output_dir", type=str, default="output/kitti_eval")
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- KITTI Calibration (2011_09_26) ----
    fx, fy = 721.5377, 721.5377
    cx, cy = 609.5593, 172.854
    baseline = 0.535164  # meters

    # KITTI Tr_velo_to_cam: rigid transform FROM velodyne TO camera (cam2).
    # Usage: X_cam = Tr_velo_to_cam @ X_velo_homogeneous
    Tr_velo_to_cam = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
        [0.0, 0.0, 0.0, 1.0],
    ])
    T_cam_velo = Tr_velo_to_cam[:3, :]  # 3x4: velo → cam

    # ---- Image paths ----
    left_dir = os.path.join(args.kitti_dir, "image_02/data")
    right_dir = os.path.join(args.kitti_dir, "image_03/data")
    lidar_dir = os.path.join(args.kitti_dir, "velodyne_points/data")

    left_files = sorted(os.listdir(left_dir))
    right_files = sorted(os.listdir(right_dir))
    lidar_files = sorted(os.listdir(lidar_dir))

    n_frames = min(len(left_files), len(right_files), len(lidar_files))
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    print(f"KITTI frames: {n_frames}")
    print(f"Calibration: fx={fx:.1f} baseline={baseline:.3f}m")
    print(f"Image size: 1242x375")

    # ---- TRT Model ----
    mh, mw = map(int, args.res.split("x"))
    onnx_path = f"weights/onnx/{args.model}/{args.res}/{args.model}_iters_{args.iters}_res_{args.res}.onnx"
    if not os.path.exists(onnx_path):
        # Try building path from available ONNX files
        candidates = [f for f in os.listdir(f"weights/onnx/{args.model}/{args.res}")
                      if f.endswith(f"iters_{args.iters}.onnx") or
                      f.endswith(f"iters_{args.iters}_res_{args.res}.onnx")]
        if candidates:
            onnx_path = f"weights/onnx/{args.model}/{args.res}/{candidates[0]}"
        else:
            print(f"ERROR: ONNX not found at {onnx_path}")
            return 1

    providers = [
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ]
    session = ort.InferenceSession(onnx_path, ort.SessionOptions(), providers=providers)
    print(f"TRT: {mh}x{mw} | {session.get_providers()[0]}")

    # ---- Resize with letterbox to preserve aspect ratio ----
    def letterbox_resize(img, target_w, target_h):
        """Resize keeping aspect ratio, pad with black to reach target size."""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_y = (target_h - new_h) // 2
        offset_x = (target_w - new_w) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        return canvas, scale, offset_x, offset_y

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

    # ---- Run on all frames ----
    all_metrics = []
    times = []

    for idx in range(n_frames):
        t0 = time.perf_counter()

        # Read images
        left = cv2.imread(os.path.join(left_dir, left_files[idx]))
        right = cv2.imread(os.path.join(right_dir, right_files[idx]))
        h_orig, w_orig = left.shape[:2]

        # Letterbox resize (preserves aspect ratio → epipolar lines intact)
        lr, lb_scale, ox, oy = letterbox_resize(left, mw, mh)
        rr, _, _, _ = letterbox_resize(right, mw, mh)

        # TRT inference
        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]  # (mh, mw)

        # Extract valid region (non-padded)
        crop_h = int(h_orig * lb_scale)
        crop_w = int(w_orig * lb_scale)
        disp_crop = disp[oy:oy+crop_h, ox:ox+crop_w]

        # Scale disparity FIRST (before resize to preserve peaks)
        # disp scales linearly with image width
        disp_crop_scaled = disp_crop * (w_orig / crop_w)

        # Then resize to original resolution
        disp_full = cv2.resize(disp_crop_scaled, (w_orig, h_orig),
                               interpolation=cv2.INTER_LINEAR)

        # Depth
        depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Compare with LiDAR ----
        lidar_pts = load_kitti_lidar(os.path.join(lidar_dir, lidar_files[idx]))
        cam_pts = project_lidar_to_camera(lidar_pts, T_cam_velo)

        # Project LiDAR to image plane
        xs = (fx * cam_pts[:, 0] / cam_pts[:, 2] + cx).astype(int)
        ys = (fy * cam_pts[:, 1] / cam_pts[:, 2] + cy).astype(int)

        valid_lidar = (cam_pts[:, 2] > 0.1) & (cam_pts[:, 2] < args.zfar) & \
                      (xs >= 0) & (xs < w_orig) & (ys >= 0) & (ys < h_orig)
        xs, ys = xs[valid_lidar], ys[valid_lidar]
        depth_gt = cam_pts[valid_lidar, 2]

        # Build GT depth map (for sparse LiDAR points)
        depth_gt_map = np.zeros_like(depth)
        for x, y, d in zip(xs, ys, depth_gt):
            if depth_gt_map[y, x] == 0 or d < depth_gt_map[y, x]:
                depth_gt_map[y, x] = d
        # GT LiDAR sparse depth map (0 elsewhere)

        # Evaluate
        mask = (depth > 0.1) & (depth < args.zfar) & (depth_gt_map > 0.1)
        metrics = evaluate_depth(depth, depth_gt_map, mask)
        if metrics:
            all_metrics.append(metrics)

        # ---- Visualization (every N frames) ----
        if idx % 5 == 0:
            # Disparity colormap
            v = disp_full > 0.5
            lo = np.percentile(disp_full[v], 2) if v.any() else 0
            hi = np.percentile(disp_full[v], 98) if v.any() else 1
            disp_vis = vis_disparity(disp_full, min_val=lo, max_val=hi)

            # Depth colormap
            vd = depth > 0
            dlo = 0
            dhi = min(80, np.percentile(depth[vd], 95)) if vd.any() else 80
            depth_vis = vis_disparity(depth, min_val=dlo, max_val=dhi, color_map=cv2.COLORMAP_JET)

            # LiDAR GT depth colormap (same range)
            gt_vis = vis_disparity(depth_gt_map, min_val=dlo, max_val=dhi, color_map=cv2.COLORMAP_JET)

            # Error map
            error_map = np.zeros_like(depth)
            error_map[mask] = np.abs(depth[mask] - depth_gt_map[mask])
            err_max = min(10, np.percentile(error_map[mask], 95)) if mask.any() else 10
            err_vis = vis_disparity(error_map, min_val=0, max_val=err_max, color_map=cv2.COLORMAP_HOT)

            # Composite
            gap = np.zeros((h_orig, 4, 3), dtype=np.uint8)
            top = np.hstack([left, disp_vis, gap, depth_vis])
            bot = np.hstack([gt_vis, err_vis, gap,
                             np.zeros_like(depth_vis)])  # placeholder for LiDAR overlay
            composite = np.vstack([top, bot])

            cv2.putText(composite, f"Left Image | Disparity | Depth", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(composite, f"LiDAR GT | Error (|pred-GT|) | Frame {idx}",
                       (10, h_orig+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(composite, f"RMSE={metrics.get('RMSE',0):.2f}m a1={metrics.get('a1',0):.3f}",
                       (10, h_orig+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            out_path = os.path.join(args.output_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(out_path, composite)

        if idx % 10 == 0:
            ms = np.mean(times[-10:]) * 1000 if times else 0
            rmse = metrics.get("RMSE", 0) if metrics else 0
            print(f"  [{idx:3d}/{n_frames}] {ms:.0f}ms | RMSE={rmse:.3f}m | "
                  f"a1={metrics.get('a1',0):.3f}", flush=True)

    # ---- Summary ----
    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n{'='*50}")
        print(f"KITTI Evaluation Results ({len(all_metrics)} frames)")
        print(f"  RMSE:  {avg['RMSE']:.4f} m")
        print(f"  MAE:   {avg['MAE']:.4f} m")
        print(f"  Rel:   {avg['Rel']:.4f}")
        print(f"  δ<1.25: {avg['a1']:.4f}")
        print(f"  δ<1.25²: {avg['a2']:.4f}")
        print(f"  δ<1.25³: {avg['a3']:.4f}")
        print(f"  Avg points: {avg['pixels']:.0f}")
        avg_ms = np.mean(times) * 1000
        print(f"  Avg time: {avg_ms:.1f}ms ({1000/avg_ms:.1f} FPS)")
        print(f"{'='*50}")

        # Save metrics
        np.savez(os.path.join(args.output_dir, "kitti_metrics.npz"),
                 avg_metrics=avg, per_frame=all_metrics)


if __name__ == "__main__":
    sys.exit(main())
