#!/usr/bin/env python3
"""
3D obstacle detection from FFS stereo depth on KITTI.

Pipeline: Depth → Point Cloud → RANSAC ground removal → DBSCAN clustering → 3D boxes

Usage:
    python scripts/kitti_3d_detect.py --kitti_dir ../libSGM/data/2011_09_26_drive_0001_sync
"""

import os, sys, cv2, numpy as np, argparse, time, onnxruntime as ort
from sklearn.cluster import DBSCAN

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


def depth_to_pointcloud(depth, fx, fy, cx, cy, step=2, zmax=80):
    """Back-project depth to 3D point cloud (camera coords)."""
    h, w = depth.shape
    vs, us = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing='ij')
    zs = depth[vs.ravel(), us.ravel()]
    valid = (zs > 0.5) & (zs < zmax)
    zs, vs2, us2 = zs[valid], vs.ravel()[valid], us.ravel()[valid]
    xs = (us2 - cx) * zs / fx
    ys = (vs2 - cy) * zs / fy
    return np.column_stack([xs, ys, zs]), vs2, us2


def remove_ground(pts_3d, distance_threshold=0.15, max_iterations=100):
    """RANSAC plane fitting to find and remove ground points.
    Returns non-ground points and the ground plane normal."""
    if len(pts_3d) < 100:
        return pts_3d, None

    # RANSAC on plane: aX + bY + cZ + d = 0
    best_inliers = []
    best_normal = None
    for _ in range(max_iterations):
        sample = pts_3d[np.random.choice(len(pts_3d), 3, replace=False)]
        p1, p2, p3 = sample
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        # Plane: normal · (x,y,z) + d = 0
        d_plane = -np.dot(normal, p1)
        distances = np.abs(pts_3d @ normal + d_plane)
        inliers = np.where(distances < distance_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal

    if len(best_inliers) < 100:
        return pts_3d, None

    mask = np.ones(len(pts_3d), dtype=bool)
    mask[best_inliers] = False
    return pts_3d[mask], best_normal


def cluster_objects(pts_3d, eps=0.5, min_samples=20):
    """DBSCAN clustering to separate individual obstacles."""
    if len(pts_3d) < min_samples:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_3d)
    labels = clustering.labels_
    clusters = []
    for label in np.unique(labels):
        if label < 0:  # noise
            continue
        cluster_pts = pts_3d[labels == label]
        if len(cluster_pts) < min_samples:
            continue
        # Filter out extremely tall/thin clusters (noise)
        h = cluster_pts[:, 1].max() - cluster_pts[:, 1].min()
        w = cluster_pts[:, 0].max() - cluster_pts[:, 0].min()
        d = cluster_pts[:, 2].max() - cluster_pts[:, 2].min()
        if h > 10 or w < 0.1:  # too tall = artifact, too thin = artifact
            continue
        clusters.append(cluster_pts)
    return clusters


def compute_3d_box(pts):
    """Compute axis-aligned 3D bounding box. Returns 8 corners."""
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_min, y_max, z_min], [x_max, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_min, y_max, z_max], [x_max, y_max, z_max],
    ])
    center = np.array([(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2])
    dims = np.array([x_max-x_min, y_max-y_min, z_max-z_min])
    return corners, center, dims


def project_3d_box(corners, fx, fy, cx, cy):
    """Project 8 box corners to image plane. Returns (8,2) pixel coords."""
    pts_2d = []
    for pt in corners:
        if pt[2] < 0.1:
            return None
        u = fx * pt[0] / pt[2] + cx
        v = fy * pt[1] / pt[2] + cy
        pts_2d.append([u, v])
    return np.array(pts_2d)


def draw_box(image, corners_2d, distance, dims, color=(0, 255, 0)):
    """Draw 3D box projection on image."""
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for i, j in edges:
        pt1 = tuple(corners_2d[i].astype(int))
        pt2 = tuple(corners_2d[j].astype(int))
        cv2.line(image, pt1, pt2, color, 1)

    # Center + distance label
    cx_box = int(corners_2d[:, 0].mean())
    cy_box = int(corners_2d[:, 1].min()) - 5
    label = f"{distance:.1f}m {dims[0]*100:.0f}x{dims[2]*100:.0f}cm"
    cv2.putText(image, label, (cx_box - 40, max(cy_box, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str,
                        default="../libSGM/data/2011_09_26_drive_0001_sync")
    parser.add_argument("--model", type=str, default="20_26_39")
    parser.add_argument("--res", type=str, default="576x960")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--zfar", type=float, default=60)
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

    left_dir = os.path.join(args.kitti_dir, "image_02/data")
    right_dir = os.path.join(args.kitti_dir, "image_03/data")
    files = sorted(os.listdir(left_dir))
    print(f"KITTI 3D Detection | {len(files)} frames | TRT {session.get_providers()[0]}")

    buf_l = np.empty((mh, mw, 3), dtype=np.float32)
    buf_r = np.empty((mh, mw, 3), dtype=np.float32)
    def norm(img, buf):
        np.multiply(img, 1.0/255.0, out=buf)
        np.subtract(buf, MEAN, out=buf); np.divide(buf, STD, out=buf)
        return np.ascontiguousarray(buf.transpose(2,0,1)[None])

    # Warmup
    d = np.random.randint(0, 256, (mh, mw, 3), dtype=np.uint8)
    dw = norm(d, buf_l.copy())
    for _ in range(3):
        session.run(['disparity'], {'left_image': dw, 'right_image': dw})

    frame_idx, times = 0, []
    disp_smooth = None
    cv2.namedWindow("3D Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3D Obstacle Detection", 1400, 450)

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

        # ---- 3D Detection Pipeline ----
        detections = []
        # Step 1: Depth → Point Cloud (step=2 for speed)
        pts_3d, vs, us = depth_to_pointcloud(depth, FX, FY, CX, CY, step=2, zmax=args.zfar)

        # Step 2: Remove ground
        objects_3d, ground_normal = remove_ground(pts_3d)

        # Step 3: Cluster
        clusters = cluster_objects(objects_3d, eps=0.8, min_samples=30)

        # Step 4: 3D boxes → 2D projection
        vis_img = left.copy()
        for cluster in clusters:
            corners, center, dims = compute_3d_box(cluster)
            corners_2d = project_3d_box(corners, FX, FY, CX, CY)
            if corners_2d is not None:
                dist = np.linalg.norm(center)
                draw_box(vis_img, corners_2d, dist, dims)
                detections.append({'distance': dist, 'center': center, 'dims': dims, 'pts': len(cluster)})

        dt = time.perf_counter() - t0
        times.append(dt)

        # ---- Display ----
        vd = depth > 0
        d_max = min(args.zfar, np.percentile(depth[vd], 95)) if vd.any() else args.zfar
        depth_vis = vis_disparity(depth, min_val=0, max_val=d_max, color_map=cv2.COLORMAP_JET)
        v = disp_smooth > 0.5
        lo = np.percentile(disp_smooth[v], 2) if v.any() else 0
        hi = np.percentile(disp_smooth[v], 98) if v.any() else 1
        disp_vis = vis_disparity(disp_smooth, min_val=lo, max_val=hi)

        panel = np.hstack([vis_img, disp_vis, depth_vis])
        avg_ms = np.mean(times[-15:]) * 1000 if times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        info = f"KITTI 3D Detect | {fps:.0f}FPS | {len(detections)} objects"
        cv2.putText(panel, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        for i, det in enumerate(detections[:5]):
            cv2.putText(panel, f"  obj{i}: {det['distance']:.1f}m {det['dims'][0]*100:.0f}x{det['dims'][2]*100:.0f}cm",
                        (10, 45 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("3D Obstacle Detection", panel)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cv2.destroyAllWindows()
    avg_ms = np.mean(times[3:]) * 1000 if len(times) > 3 else 0
    print(f"\nDone. {frame_idx} frames, avg {avg_ms:.0f}ms ({1000/avg_ms:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
