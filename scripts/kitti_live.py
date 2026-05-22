#!/usr/bin/env python3
"""
KITTI sequence live depth + BEV (Bird's Eye View) visualization.

Left | Disparity | Depth
BEV  | FreeSpace | HeightMap

Usage:
    python scripts/kitti_live.py --kitti_dir ../libSGM/data/2011_09_26_drive_0001_sync
"""

import os, sys, cv2, numpy as np, argparse, time, onnxruntime as ort

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
FX, FY, CX, CY = 721.5377, 721.5377, 609.5593, 172.854
BASELINE = 0.535164


def letterbox(img, tw, th):
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas, s, ox, oy


def fast_bev(depth, fx, fy, cx, cy, grid_size=400, z_max=50, x_range=25, cam_height=1.65):
    """Back-project depth to BEV. Only keep ground-level points.
    Color: ground surface = green, near-ground objects = yellow/red.
    """
    h, w = depth.shape
    vs, us = np.meshgrid(np.arange(0, h, 2), np.arange(0, w, 2), indexing='ij')
    vs, us = vs.ravel(), us.ravel()
    zs = depth[vs, us]

    valid = (zs > 0.5) & (zs < z_max)
    zs, vs, us = zs[valid], vs[valid], us[valid]

    # Camera 3D: X=right, Y=down, Z=forward
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy

    # Height above ground: camera is cam_height above ground plane
    # Ground plane in camera coords: Y = cam_height (since Y is down)
    # height_above_ground = cam_height - Y
    # (positive = above ground, negative = below ground/road)
    height_ag = cam_height - ys

    # Only keep points near ground level (road, cars, curbs, etc.)
    ground_mask = (height_ag > -0.5) & (height_ag < 3.0)
    zs, xs, height_ag = zs[ground_mask], xs[ground_mask], height_ag[ground_mask]

    if len(zs) < 100:
        return (np.zeros((grid_size, grid_size, 3), dtype=np.uint8),
                np.zeros((grid_size, grid_size, 3), dtype=np.uint8),
                np.zeros((grid_size, grid_size), dtype=bool))

    # BEV grid: x→column (left-right), z→row (near-far)
    xi = ((xs / x_range + 1) * 0.5 * grid_size).astype(int)
    zi = (zs / z_max * grid_size).astype(int)

    cell_ok = (xi >= 0) & (xi < grid_size) & (zi >= 0) & (zi < grid_size)
    xi, zi, height_ag = xi[cell_ok], zi[cell_ok], height_ag[cell_ok]

    height_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.int32)
    np.add.at(count, (zi, xi), 1)
    np.add.at(height_sum, (zi, xi), height_ag)

    mask = count > 0
    avg_h = np.divide(height_sum, count, where=mask)

    # Color: ground(0-0.3m)=green, low obj(0.3-1m)=yellow, high obj(1m+)=red
    bev = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    h_clip = np.clip(avg_h[mask], 0, 2.0)
    t = h_clip / 2.0  # 0→ground, 1→tall
    r = (t * 255).astype(np.uint8)
    g = ((1 - t) * 200 + 55).astype(np.uint8)
    b_chan = np.zeros_like(r)
    idx = np.where(mask)
    bev[idx[0], idx[1], 0] = b_chan
    bev[idx[0], idx[1], 1] = g
    bev[idx[0], idx[1], 2] = r

    # Freespace: occupied cells = grey
    freespace = np.zeros_like(bev)
    freespace[idx[0], idx[1]] = (60, 60, 60)

    return bev, freespace, mask


def draw_bev_grid(img, z_max, x_range, grid_size):
    """Add grid lines and labels to BEV image."""
    # Grid every 10m
    for z_m in range(0, int(z_max) + 1, 10):
        zi = int(z_m / z_max * grid_size)
        if zi < grid_size:
            cv2.line(img, (0, zi), (grid_size - 5, zi), (60, 60, 60), 1)
            cv2.putText(img, f"{z_m}m", (2, zi - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    # Lateral lines
    for x_m in range(-int(x_range), int(x_range) + 1, 10):
        xi = int((x_m / x_range + 1) * 0.5 * grid_size)
        if 0 <= xi < grid_size:
            cv2.line(img, (xi, 0), (xi, grid_size - 5), (60, 60, 60), 1)
    # Car position marker
    cx_bev = grid_size // 2
    cv2.rectangle(img, (cx_bev - 3, grid_size - 8),
                  (cx_bev + 3, grid_size - 1), (0, 255, 0), -1)
    cv2.putText(img, "Ego", (cx_bev + 5, grid_size - 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str,
                        default="../libSGM/data/2011_09_26_drive_0001_sync")
    parser.add_argument("--model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="576x960")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--zfar", type=float, default=80)
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    mh, mw = map(int, args.res.split("x"))
    onnx = f"weights/onnx/{args.model}/{args.res}/{args.model}_iters_{args.iters}_res_{args.res}.onnx"
    session = ort.InferenceSession(onnx, ort.SessionOptions(), providers=[
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ])
    print(f"TRT: {session.get_providers()[0]} | {mh}x{mw}")

    left_dir = os.path.join(args.kitti_dir, "image_02/data")
    right_dir = os.path.join(args.kitti_dir, "image_03/data")
    files = sorted(os.listdir(left_dir))
    print(f"KITTI: {len(files)} frames | q=quit t=smooth")

    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)

    def norm(img, buf):
        np.multiply(img, 1.0 / 255.0, out=buf)
        np.subtract(buf, MEAN, out=buf)
        np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2, 0, 1)[None])

    # Warmup
    d = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
    dw = norm(d, buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    frame_idx = 0
    times = []
    disp_smooth = None
    disp_lo, disp_hi = None, None
    bev_size = 400
    z_max = 50
    x_range = 25

    cv2.namedWindow("KITTI + Depth + BEV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("KITTI + Depth + BEV", 1800, 600)

    while frame_idx < len(files):
        t0 = time.perf_counter()
        left = cv2.imread(os.path.join(left_dir, files[frame_idx]))
        right = cv2.imread(os.path.join(right_dir, files[frame_idx]))
        h_orig, w_orig = left.shape[:2]
        lr, s, ox, oy = letterbox(left, mw, mh)
        rr, _, _, _ = letterbox(right, mw, mh)

        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]
        nw, nh = int(w_orig * s), int(h_orig * s)
        disp = disp[oy:oy + nh, ox:ox + nw] * (w_orig / nw)
        disp_full = cv2.resize(disp, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        if disp_smooth is not None:
            disp_smooth = args.temporal * disp_full + (1 - args.temporal) * disp_smooth
        else:
            disp_smooth = disp_full.copy()

        depth = np.where(disp_smooth > 0.5, FX * BASELINE / disp_smooth, 0)

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- BEV computation ----
        bev, freespace, _ = fast_bev(depth, FX, FY, CX, CY,
                                     grid_size=bev_size, z_max=z_max, x_range=x_range)
        draw_bev_grid(bev, z_max, x_range, bev_size)
        draw_bev_grid(freespace, z_max, x_range, bev_size)

        # ---- Depth/Disparity visualization ----
        v = disp_smooth > 0.5
        if v.any():
            if disp_lo is None:
                disp_lo = np.percentile(disp_smooth[v], 2)
                disp_hi = np.percentile(disp_smooth[v], 98)
        disp_vis = vis_disparity(disp_smooth, min_val=disp_lo or 0, max_val=disp_hi or 1)

        vd = depth > 0
        d_max = min(args.zfar, np.percentile(depth[vd], 95)) if vd.any() else args.zfar
        depth_vis = vis_disparity(depth, min_val=0, max_val=d_max, color_map=cv2.COLORMAP_JET)

        # ---- Compose layout ----
        # Top row: Left | Disparity | Depth → all same height h_orig
        top_h = h_orig
        top = np.hstack([left, disp_vis, depth_vis])

        # Bottom row: BEV height_map | BEV freespace | padded
        # Scale BEV to match top row height
        bev_scaled = cv2.resize(bev, (int(bev_size * top_h / bev_size),
                                       int(bev_size * top_h / bev_size)))
        fs_scaled = cv2.resize(freespace, (bev_scaled.shape[1], bev_scaled.shape[0]))
        bot = np.hstack([bev_scaled, fs_scaled,
                         np.zeros((top_h, top.shape[1] - 2 * bev_scaled.shape[1], 3),
                                  dtype=np.uint8)])
        display = np.vstack([top, bot])

        avg_ms = np.mean(times[-15:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        d_mean = depth[vd].mean() if vd.any() else 0
        info = (f"KITTI | {fps:.0f}FPS | depth:{d_mean:.1f}m | BEV {z_max}x{x_range*2}m | f{frame_idx}/{len(files)}")
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(display, "Height Map (green=gnd red=high)", (10, top_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, "Occupancy", (bev_scaled.shape[1] + 10, top_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("KITTI + Depth + BEV", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            if disp_smooth is not None:
                disp_smooth = None
            else:
                disp_smooth = disp_full.copy()

        frame_idx += 1

    cv2.destroyAllWindows()
    avg_ms = np.mean(times[3:]) * 1000 if len(times) > 3 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
