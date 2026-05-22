#!/usr/bin/env python3
"""
Real-time stereo depth estimation with TensorRT acceleration.

Pre-built ONNX models are compiled into TRT engines via ONNX Runtime.
Preprocessing uses fused in-place normalization to minimize CPU overhead.

Usage:
    python scripts/realtime_depth_trt.py --model 20_26_39 --res 320x736
    python scripts/realtime_depth_trt.py --images video_frames
"""

import os, sys
import cv2
import numpy as np
import argparse, pickle, time
import onnxruntime as ort

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--images", type=str, default="")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--model", type=str, default="23_36_37",
                        choices=["23_36_37", "20_30_48", "20_26_39"])
    parser.add_argument("--res", type=str, default="576x960",
                        choices=["576x960", "320x736", "640x480"])
    parser.add_argument("--iters", type=int, default=4, choices=[4, 8])
    parser.add_argument("--output", type=str, default="output/")
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--zfar", type=float, default=10)
    parser.add_argument("--temporal", type=float, default=0.3)
    parser.add_argument("--no_rectify", action="store_true")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show_pc", action="store_true")
    parser.add_argument("--pc_step", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- ONNX model ----
    res_dir = args.res
    onnx_dir = f"weights/onnx/{args.model}/{res_dir}"
    onnx_path = f"{onnx_dir}/{args.model}_iters_{args.iters}_res_{res_dir}.onnx"
    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found"); return 1

    # ---- Calibration ----
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline = calib["baseline"]
    K_depth = calib["P1"].copy()
    fx = K_depth[0, 0]
    print(f"fx={fx:.0f} baseline={baseline*1000:.0f}mm")

    # ---- Rectification maps ----
    maps_file = os.path.join(os.path.dirname(args.calib), "rectification_maps.npz")
    use_rectify = False
    if os.path.exists(maps_file):
        rect_maps = np.load(maps_file)
        map_lx, map_ly = rect_maps["map_lx"], rect_maps["map_ly"]
        map_rx, map_ry = rect_maps["map_rx"], rect_maps["map_ry"]
        use_rectify = True
    if args.no_rectify:
        use_rectify = False
    print(f"Rectify: {'ON' if use_rectify else 'OFF'}")

    # ---- TRT session ----
    session = ort.InferenceSession(onnx_path, providers=[
        ('TensorrtExecutionProvider', {
            'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': args.cache_dir,
        }), 'CUDAExecutionProvider', 'CPUExecutionProvider',
    ])
    print(f"Provider: {session.get_providers()[0]}")

    _, _, model_h, model_w = session.get_inputs()[0].shape
    disp_scale = args.half_w / float(model_w)

    # ---- Source ----
    images_dir, image_paths = None, []
    vid_fps = 30
    if args.images:
        images_dir = args.images
        image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)
                              if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    else:
        cap = cv2.VideoCapture(args.video)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Temporal smooth: {args.temporal}")

    # ---- Pre-allocated normalize buffers ----
    mean = np.float32([0.485, 0.456, 0.406])
    std  = np.float32([0.229, 0.224, 0.225])
    scale = np.float32(1.0 / 255.0)
    buf_l = np.empty((model_h, model_w, 3), dtype=np.float32)
    buf_r = np.empty((model_h, model_w, 3), dtype=np.float32)

    def norm(img_uint8, buf):
        # Fused in-place normalize: avoids intermediate array allocations
        np.multiply(img_uint8, scale, out=buf)
        np.subtract(buf, mean, out=buf)
        np.divide(buf, std, out=buf)
        return np.ascontiguousarray(buf.transpose(2, 0, 1)[None])

    # Warmup
    dummy = np.random.randint(0, 256, (model_h, model_w, 3), dtype=np.uint8)
    dw = norm(dummy, buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    # ---- Point cloud (optional) ----
    pc_vis = None
    save_pc_dir = None
    K_pc = K_depth.copy()
    if args.show_pc:
        try:
            pc_vis = o3d.visualization.Visualizer()
            pc_vis.create_window(window_name="Point Cloud", width=800, height=600)
            if pc_vis.get_view_control() is None:
                raise RuntimeError("GL context broken")
            print(f"Point cloud window OK (step={args.pc_step})")
        except Exception as e:
            print(f"Window unavailable, saving PLY instead: {e}")
            save_pc_dir = os.path.join(args.output, "pc_frames")
            os.makedirs(save_pc_dir, exist_ok=True)
            pc_vis = None

    # ---- State ----
    writer, disp_smooth = None, None
    smooth_enabled = args.temporal > 0
    alpha = args.temporal
    frame_idx = 0
    times = []
    disp_lo, disp_hi = None, None
    depth_lo, depth_hi = None, None

    while True:
        # Read frame
        if images_dir:
            if frame_idx >= len(image_paths): break
            frame = cv2.imread(image_paths[frame_idx])
            if frame is None: break
        else:
            ret, frame = cap.read()
            if not ret: break
        t0 = time.perf_counter()

        # Split concatenated stereo pair
        left_raw  = frame[:, :args.half_w]
        right_raw = frame[:, args.half_w:]

        # Rectify
        if use_rectify:
            left_raw = cv2.remap(left_raw, map_lx, map_ly, cv2.INTER_LINEAR)
            right_raw = cv2.remap(right_raw, map_rx, map_ry, cv2.INTER_LINEAR)

        # Resize to model input
        lr = cv2.resize(left_raw,  (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        rr = cv2.resize(right_raw, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

        # Normalize + TRT inference
        ln = norm(lr, buf_l)
        rn = norm(rr, buf_r)
        out = session.run(['disparity'], {'left_image': ln, 'right_image': rn})
        disp = out[0][0, 0]

        # Temporal smooth (EMA)
        if smooth_enabled and disp_smooth is not None:
            disp_smooth = alpha * disp + (1 - alpha) * disp_smooth
        else:
            disp_smooth = disp.copy()

        # Upsample disparity and convert to depth
        disp_full = cv2.resize(disp_smooth, (args.half_w, left_raw.shape[0]),
                               interpolation=cv2.INTER_LINEAR) * disp_scale
        depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        dt = time.perf_counter() - t0
        times.append(dt)

        # Point cloud generation (every 2nd frame)
        if args.show_pc and frame_idx % 2 == 0:
            step = args.pc_step
            depth_pc = depth[::step, ::step]
            color_pc = cv2.cvtColor(left_raw[::step, ::step], cv2.COLOR_BGR2RGB)
            K_scaled = K_pc.copy()
            K_scaled[:2] /= step
            pts = depth2xyzmap(depth_pc, K_scaled)
            pts_flat = pts.reshape(-1, 3)
            valid = (pts_flat[:, 2] > 0.1) & (pts_flat[:, 2] < args.zfar)
            pcd = toOpen3dCloud(pts_flat[valid], color_pc.reshape(-1, 3)[valid])
            if pc_vis is not None:
                if frame_idx == 0:
                    pc_vis.add_geometry(pcd)
                    ctr = pc_vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                else:
                    pc_vis.remove_geometry(pcd)
                    pc_vis.add_geometry(pcd)
                pc_vis.poll_events()
                pc_vis.update_renderer()
            elif save_pc_dir:
                o3d.io.write_point_cloud(f"{save_pc_dir}/frame_{frame_idx:06d}.ply", pcd)

        # Visualization
        valid = disp_full > 0.5
        if valid.any():
            if disp_lo is None and frame_idx >= 8:
                disp_lo = np.percentile(disp_full[valid], 2)
                disp_hi = np.percentile(disp_full[valid], 98)
            elif disp_lo is None:
                disp_lo, disp_hi = np.percentile(disp_full[valid], 2), np.percentile(disp_full[valid], 98)

        disp_vis = vis_disparity(disp_full, min_val=disp_lo or 0, max_val=disp_hi or 1,
                                 color_map=cv2.COLORMAP_TURBO)

        depth_clipped = np.where(depth > args.zfar, 0, depth)
        valid_d = depth_clipped > 0
        if valid_d.any():
            if depth_lo is None and frame_idx >= 8:
                depth_lo = np.percentile(depth_clipped[valid_d], 2)
                depth_hi = np.percentile(depth_clipped[valid_d], 98)
            elif depth_lo is None:
                depth_lo, depth_hi = np.percentile(depth_clipped[valid_d], 2), np.percentile(depth_clipped[valid_d], 98)

        depth_vis = vis_disparity(depth_clipped, min_val=depth_lo or 0, max_val=depth_hi or 1,
                                  color_map=cv2.COLORMAP_JET)

        # Compose display panel
        top = np.hstack([left_raw, right_raw, disp_vis])
        s = disp_vis.shape[0] / max(depth_vis.shape[0], 1)
        depth_vis_s = cv2.resize(depth_vis, None, fx=s, fy=s)
        pad_w = top.shape[1] - depth_vis_s.shape[1]
        bottom = np.hstack([depth_vis_s, np.zeros((depth_vis_s.shape[0], pad_w, 3), dtype=np.uint8)])
        display = np.vstack([top, bottom])

        avg_ms = np.mean(times[-30:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        d_mean = depth[valid_d].mean() if valid_d.any() else 0
        info = (f"TRT {args.model} | {fps:.1f} FPS | depth:{d_mean:.2f}m | "
                f"{'SMOOTH' if smooth_enabled else 'RAW'}")
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if not args.no_display:
            if frame_idx == 0:
                cv2.namedWindow("FFS TRT", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("FFS TRT", 1600, 600)
            cv2.imshow("FFS TRT", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord("t"):
                smooth_enabled = not smooth_enabled
                if not smooth_enabled: disp_smooth = None
            elif key in (ord("+"), ord("=")): alpha = min(0.9, alpha + 0.1)
            elif key == ord("-"): alpha = max(0.0, alpha - 0.1)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(
                    os.path.join(args.output, "depth_trt.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), vid_fps,
                    (display.shape[1], display.shape[0]))
            writer.write(display)

        if args.no_display and frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:3d}: {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)", flush=True)
        frame_idx += 1

    if not images_dir: cap.release()
    if writer: writer.release()
    if pc_vis: pc_vis.destroy_window()
    if save_pc_dir: print(f"Point cloud frames saved to {save_pc_dir}/")
    cv2.destroyAllWindows()

    avg_ms = np.mean(times[5:]) * 1000 if len(times) > 5 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
