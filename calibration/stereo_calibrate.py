"""
Zhang's stereo calibration for 8x4 chessboard (9x5 squares).
Input: concatenated left-right images in calibration/
Output: K.txt (for FFS), calib_data.pkl, rectification_maps.npz
"""

import cv2
import numpy as np
import argparse
import glob
import os
import pickle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_dir", type=str, default="calibration/")
    parser.add_argument("--pattern", type=str, default="8,4",
                        help="Inner corners: cols,rows")
    parser.add_argument("--square_size", type=float, default=0.020,
                        help="Square size in meters")
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--out_dir", type=str, default="calibration/calib_results/")
    args = parser.parse_args()

    pw, ph = map(int, args.pattern.split(","))
    pattern = (pw, ph)
    objp = np.zeros((ph * pw, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pw, 0:ph].T.reshape(-1, 2) * args.square_size

    os.makedirs(args.out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.calib_dir, "*.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(args.calib_dir, "*.png")))
    print(f"Found {len(img_paths)} images")

    objpts_l, imgpts_l = [], []
    objpts_r, imgpts_r = [], []
    stereo_obj, stereo_l, stereo_r = [], [], []
    h, w = None, None
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        if h is None:
            h, fw = img.shape[:2]
            w = args.half_w

        gray_l = cv2.cvtColor(img[:, :w], cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img[:, w:], cv2.COLOR_BGR2GRAY)

        # Zhang's method uses classic findChessboardCorners
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern, None)

        if not ret_l and not ret_r:
            # Try CLAHE enhancement as fallback
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ret_l, corners_l = cv2.findChessboardCorners(clahe.apply(gray_l), pattern, None)
            ret_r, corners_r = cv2.findChessboardCorners(clahe.apply(gray_r), pattern, None)

        status = []
        if ret_l:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            objpts_l.append(objp)
            imgpts_l.append(corners_l)
            status.append("L")
        if ret_r:
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)
            objpts_r.append(objp)
            imgpts_r.append(corners_r)
            status.append("R")
        if ret_l and ret_r:
            stereo_obj.append(objp)
            stereo_l.append(corners_l)
            stereo_r.append(corners_r)

        s = "+".join(status) if status else "NONE"
        print(f"  [{i:2d}] {os.path.basename(p)}: {s}")

    print(f"\nDetections: L={len(objpts_l)}, R={len(objpts_r)}, stereo pairs={len(stereo_obj)}")

    # Monocular calibration
    calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
    calib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    print("\n--- Left camera ---")
    rms_l, Kl, Dl, _, _ = cv2.calibrateCamera(objpts_l, imgpts_l, (w, h), None, None)
    print(f"  RMS = {rms_l:.4f} px")
    print(f"  K = [[{Kl[0,0]:.1f}, 0, {Kl[0,2]:.1f}], [0, {Kl[1,1]:.1f}, {Kl[1,2]:.1f}]]")
    print(f"  D = {Dl.ravel()}")

    print("\n--- Right camera ---")
    rms_r, Kr, Dr, _, _ = cv2.calibrateCamera(objpts_r, imgpts_r, (w, h), None, None)
    print(f"  RMS = {rms_r:.4f} px")
    print(f"  K = [[{Kr[0,0]:.1f}, 0, {Kr[0,2]:.1f}], [0, {Kr[1,1]:.1f}, {Kr[1,2]:.1f}]]")
    print(f"  D = {Dr.ravel()}")

    # Stereo calibration
    print("\n--- Stereo calibration ---")
    rms_s, Kl2, Dl2, Kr2, Dr2, R, T, E, F = cv2.stereoCalibrate(
        stereo_obj, stereo_l, stereo_r,
        Kl, Dl, Kr, Dr, (w, h),
        criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-7),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    baseline = np.linalg.norm(T)
    print(f"  Stereo RMS = {rms_s:.4f} px")
    print(f"  Baseline = {baseline:.4f} m ({baseline*1000:.1f} mm)")
    print(f"  R:\n{R}")
    print(f"  T: {T.ravel()}")

    # Rectification
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        Kl2, Dl2, Kr2, Dr2, (w, h), R, T, alpha=0
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(Kl2, Dl2, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(Kr2, Dr2, R2, P2, (w, h), cv2.CV_32FC1)

    fx = P1[0, 0]
    cx = P1[0, 2]
    cy = P1[1, 2]
    print(f"\n  Rectified: fx={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  ROI L: {roi_l}, ROI R: {roi_r}")

    # Save
    # K.txt for FFS
    with open(os.path.join(args.out_dir, "K.txt"), "w") as f:
        f.write(f"{fx} 0.0 {cx} 0.0 {fx} {cy} 0.0 0.0 1.0\n")
        f.write(f"{baseline}\n")

    # Full calibration data
    with open(os.path.join(args.out_dir, "calib_data.pkl"), "wb") as f:
        pickle.dump({
            "K_left": Kl2, "D_left": Dl2,
            "K_right": Kr2, "D_right": Dr2,
            "R": R, "T": T, "E": E, "F": F,
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "roi_left": roi_l, "roi_right": roi_r,
            "baseline": baseline, "image_size": (w, h),
            "rms_mono_left": rms_l, "rms_mono_right": rms_r, "rms_stereo": rms_s,
        }, f)

    # Rectification maps
    np.savez_compressed(os.path.join(args.out_dir, "rectification_maps.npz"),
                        map_lx=map_lx, map_ly=map_ly,
                        map_rx=map_rx, map_ry=map_ry)

    print(f"\nSaved to {args.out_dir}/")
    print(f"  K.txt, calib_data.pkl, rectification_maps.npz")
    print(f"  fx={fx:.1f}  baseline={baseline*1000:.1f}mm  stereo_rms={rms_s:.2f}px")

    return 0


if __name__ == "__main__":
    sys.exit(main())
