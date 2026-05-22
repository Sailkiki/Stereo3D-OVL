"""
Real-time stereo depth from video using Fast-FoundationStereo.
Usage:
    python scripts/realtime_depth.py --video video/office.avi.avi
    python scripts/realtime_depth.py --scale 1.0           # full res (3 FPS)
    python scripts/realtime_depth.py --scale 0.5           # half res (13 FPS, default)
    python scripts/realtime_depth.py --scale 0.33          # third res (20+ FPS)
"""

import cv2
import numpy as np
import torch
import argparse
import pickle
import os
import sys
import time

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)

from core.foundation_stereo import FastFoundationStereo
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--model", type=str, default="weights/model_best_bp2_serialize.pth")
    parser.add_argument("--output", type=str, default="output/")
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Downsample factor for inference (0.5 = 960x540, 1.0 = 1920x1080)")
    parser.add_argument("--valid_iters", type=int, default=4, help="4=faster, 8=more accurate")
    parser.add_argument("--max_disp", type=int, default=256)
    parser.add_argument("--zfar", type=float, default=10, help="max depth (m) for colormap")
    parser.add_argument("--temporal", type=float, default=0.4,
                        help="Temporal smoothing factor (0=no smoothing, 0.5=heavy, default 0.4)")
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Calibration ----
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline = calib["baseline"]
    fx_orig = calib["K_left"][0, 0]

    maps_file = os.path.join(os.path.dirname(args.calib), "rectification_maps.npz")
    use_rectify = args.rectify and os.path.exists(maps_file)
    r_maps = None
    if use_rectify:
        maps = np.load(maps_file)
        r_maps = (maps["map_lx"], maps["map_ly"], maps["map_rx"], maps["map_ry"])

    # ---- Model ----
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    if not hasattr(model.args, "normalize"):
        model.args.normalize = True
    model.cuda().eval()

    # ---- Video ----
    cap = cv2.VideoCapture(args.video)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {n_frames} frames @ {vid_fps:.1f} fps")
    print(f"Baseline: {baseline*1000:.1f} mm  fx: {fx_orig:.0f}")
    print(f"Scale: {args.scale}  Temporal smoothing: {args.temporal}")
    print(f"Keys: q=quit  s=toggle iters(4/8)  t=toggle smoothing  r=toggle rectify")

    writer = None
    slow_mode = False
    smooth_enabled = args.temporal > 0
    frame_idx = 0
    times = []
    disp_lo, disp_hi = None, None  # stable visualization ranges
    depth_lo, depth_hi = None, None
    disp_smooth = None
    use_rectify = use_rectify

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()

        left_raw = frame[:, :args.half_w]
        right_raw = frame[:, args.half_w:]

        if use_rectify and r_maps is not None:
            left = cv2.remap(left_raw, *r_maps[:2], cv2.INTER_LINEAR)
            right = cv2.remap(right_raw, *r_maps[2:], cv2.INTER_LINEAR)
        else:
            left, right = left_raw, right_raw

        h_orig, w_orig = left.shape[:2]

        # Downsample for inference
        if args.scale < 1.0:
            w_infer = int(w_orig * args.scale)
            h_infer = int(h_orig * args.scale)
            left_infer = cv2.resize(left, (w_infer, h_infer), interpolation=cv2.INTER_LINEAR)
            right_infer = cv2.resize(right, (w_infer, h_infer), interpolation=cv2.INTER_LINEAR)
        else:
            left_infer, right_infer = left, right

        h_infer, w_infer = left_infer.shape[:2]

        # Inference
        img0 = torch.as_tensor(left_infer).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_infer).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        iters = 8 if slow_mode else args.valid_iters
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0_p, img1_p, iters=iters, test_mode=True,
                                 optimize_build_volume="pytorch1")
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h_infer, w_infer).clip(0, None)

        # Temporal smoothing (EMA)
        if smooth_enabled and disp_smooth is not None:
            alpha = args.temporal
            disp_smooth = alpha * disp + (1 - alpha) * disp_smooth
        else:
            disp_smooth = disp.copy()

        # Upsample disparity to original resolution
        if args.scale < 1.0:
            disp_full = cv2.resize(disp_smooth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            disp_full *= (1.0 / args.scale)  # scale disparity back
        else:
            disp_full = disp_smooth

        # Depth from disparity (use original K, scaled for inference resolution)
        fx = fx_orig * args.scale if args.scale < 1.0 else fx_orig
        depth = np.where(disp_full > 0.5, fx_orig * baseline / disp_full, 0)

        dt = time.time() - t0
        times.append(dt)

        # ---- Visualization with stable ranges ----
        # Lock ranges after first few frames to prevent color flickering
        valid = disp_full > 0.5
        if valid.any():
            if disp_lo is None and frame_idx >= 5:
                disp_lo = np.percentile(disp_full[valid], 2)
                disp_hi = np.percentile(disp_full[valid], 98)
            elif disp_lo is None:
                disp_lo, disp_hi = np.percentile(disp_full[valid], 2), np.percentile(disp_full[valid], 98)
        disp_vis = vis_disparity(disp_full, min_val=disp_lo, max_val=disp_hi, color_map=cv2.COLORMAP_TURBO)

        depth_clipped = depth.copy()
        depth_clipped[depth > args.zfar] = 0
        valid_d = depth_clipped > 0
        if valid_d.any():
            if depth_lo is None and frame_idx >= 5:
                depth_lo = np.percentile(depth_clipped[valid_d], 2)
                depth_hi = np.percentile(depth_clipped[valid_d], 98)
            elif depth_lo is None:
                depth_lo, depth_hi = np.percentile(depth_clipped[valid_d], 2), np.percentile(depth_clipped[valid_d], 98)
        depth_vis = vis_disparity(depth_clipped, min_val=depth_lo, max_val=depth_hi, color_map=cv2.COLORMAP_JET)

        # Display: [left | right | disparity] / [depth]
        top = np.hstack([left_raw, right_raw, disp_vis])
        scale_h = disp_vis.shape[0] / max(depth_vis.shape[0], 1)
        depth_vis_s = cv2.resize(depth_vis, None, fx=scale_h, fy=scale_h)
        pad_w = top.shape[1] - depth_vis_s.shape[1]
        bottom = np.hstack([depth_vis_s,
                            np.zeros((depth_vis_s.shape[0], pad_w, 3), dtype=np.uint8)])
        display = np.vstack([top, bottom])

        avg_ms = np.mean(times[-30:]) * 1000 if times else 0
        actual_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        info = (f"FPS:{actual_fps:.1f} | scale:{args.scale} | iters:{iters} | "
                f"smooth:{'ON' if smooth_enabled else 'OFF'}")
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"depth: {depth[valid_d].mean():.2f}m",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not args.no_display:
            s = 1280 / display.shape[1]
            disp_s = cv2.resize(display, (int(display.shape[1] * s), int(display.shape[0] * s)))
            cv2.imshow("Fast-FoundationStereo | Depth", disp_s)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                slow_mode = not slow_mode
            elif key == ord("t"):
                smooth_enabled = not smooth_enabled
                if not smooth_enabled:
                    disp_smooth = None
            elif key == ord("r"):
                use_rectify = not use_rectify

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(
                    os.path.join(args.output, "depth_output.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), vid_fps,
                    (display.shape[1], display.shape[0]))
            writer.write(display)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_ms = np.mean(times) * 1000 if times else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f} ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
