"""
Fuse FFS stereo depth maps using VGGT camera poses.

Step 1 (vggt3d env):
    python PROJECT/scripts/run_vggt_poses.py --frames_dir ... --output vggt_poses.npz

Step 2 (ffs env):
    python scripts/fuse_ffs_vggt.py --poses ../PROJECT/data/vggt_poses.npz
"""

import os, sys, cv2, numpy as np, argparse, pickle, time, onnxruntime as ort
import open3d as o3d

sys.stdout.reconfigure(line_buffering=True)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)
from Utils import depth2xyzmap, toOpen3dCloud

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])
SCALE = np.float32(1.0 / 255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=str, required=True, help="VGGT poses .npz file")
    parser.add_argument("--video", type=str, default="video/office.avi.avi")
    parser.add_argument("--calib", type=str, default="calibration/calib_results/calib_data.pkl")
    parser.add_argument("--model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="576x960")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--half_w", type=int, default=1920)
    parser.add_argument("--zfar", type=float, default=5)
    parser.add_argument("--zmin", type=float, default=0.2)
    parser.add_argument("--pc_step", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=0.01, help="final voxel downsample (m)")
    parser.add_argument("--output", type=str, default="output/fused_model.ply")
    parser.add_argument("--cache_dir", type=str, default="output_trt/cache")
    args = parser.parse_args()

    # ---- Load VGGT poses ----
    vggt = np.load(args.poses)
    extrinsics = vggt["extrinsic"]   # (S, 3, 4) camera→world
    frame_indices = vggt["frame_indices"]
    print(f"VGGT poses: {len(frame_indices)} frames")

    # ---- Calibration ----
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    baseline = calib["baseline"]
    P1 = calib["P1"]
    fx, fy, cx, cy = P1[0,0], P1[1,1], P1[0,2], P1[1,2]
    maps = np.load(os.path.join(os.path.dirname(args.calib), "rectification_maps.npz"))
    map_lx, map_ly = maps["map_lx"], maps["map_ly"]
    map_rx, map_ry = maps["map_rx"], maps["map_ry"]

    # ---- TRT ----
    mh, mw = map(int, args.res.split("x"))
    onnx_path = f"weights/onnx/{args.model}/{args.res}/{args.model}_iters_{args.iters}_res_{args.res}.onnx"
    providers = [
        ('TensorrtExecutionProvider', {'device_id': 0, 'trt_fp16_enable': True,
            'trt_engine_cache_enable': True, 'trt_engine_cache_path': args.cache_dir}),
        'CUDAExecutionProvider',
    ]
    session = ort.InferenceSession(onnx_path, ort.SessionOptions(), providers=providers)
    disp_scale = args.half_w / float(mw)
    print(f"TRT: {mh}x{mw} | fx={fx:.0f} | baseline={baseline*1000:.0f}mm")

    # ---- Pre-allocated buffers ----
    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)
    def norm(img, buf):
        np.multiply(img, SCALE, out=buf)
        np.subtract(buf, MEAN, out=buf)
        np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2,0,1)[None])

    # ---- K for point cloud (scaled to step) ----
    K_s = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    K_s[:2] /= args.pc_step

    # ---- Open video ----
    cap = cv2.VideoCapture(args.video)
    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_frames} frames")

    # ---- Fuse ----
    accumulated = o3d.geometry.PointCloud()
    total_pts = 0
    processed = 0

    for pose_idx, src_frame in enumerate(frame_indices):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_frame))
        ret, frame = cap.read()
        if not ret:
            # Fallback: sequential read
            pass

        # Rectify
        left = cv2.remap(frame[:, :args.half_w], map_lx, map_ly, cv2.INTER_LINEAR)
        right = cv2.remap(frame[:, args.half_w:], map_rx, map_ry, cv2.INTER_LINEAR)

        # Resize + normalize
        lr = cv2.resize(left, (mw, mh), interpolation=cv2.INTER_LINEAR)
        rr = cv2.resize(right, (mw, mh), interpolation=cv2.INTER_LINEAR)

        # TRT inference
        out = session.run(['disparity'], {
            'left_image': norm(lr, buf_l),
            'right_image': norm(rr, buf_r),
        })
        disp = out[0][0, 0]

        # Upsample + depth
        disp_full = cv2.resize(disp, (args.half_w, left.shape[0]),
                               interpolation=cv2.INTER_LINEAR) * disp_scale
        depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0)

        # Point cloud in camera frame
        depth_pc = depth[::args.pc_step, ::args.pc_step]
        color_pc = cv2.cvtColor(left[::args.pc_step, ::args.pc_step], cv2.COLOR_BGR2RGB)
        pts_cam = depth2xyzmap(depth_pc, K_s).reshape(-1, 3)
        valid = (pts_cam[:, 2] > args.zmin) & (pts_cam[:, 2] < args.zfar)

        if valid.sum() < 100:
            continue

        pts_cam = pts_cam[valid]
        cols = color_pc.reshape(-1, 3)[valid].astype(np.float64) / 255.0

        # Transform camera→world via VGGT extrinsic
        ext = extrinsics[pose_idx]  # (3, 4)
        R_c2w = ext[:3, :3]
        t_c2w = ext[:3, 3]
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w

        # Accumulate
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        accumulated += pcd
        total_pts += len(pts_world)
        processed += 1

        if processed % 5 == 0:
            print(f"  Frame {src_frame} ({processed}/{len(frame_indices)}): "
                  f"{len(pts_world)} pts | total: {total_pts}", flush=True)

    cap.release()

    # Voxel downsample to remove overlap
    print(f"Voxel downsampling ({args.voxel}m)...")
    accumulated = accumulated.voxel_down_sample(args.voxel)
    final_pts = len(accumulated.points)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    o3d.io.write_point_cloud(args.output, accumulated)
    print(f"Saved: {args.output} ({final_pts} pts from {processed} frames)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
