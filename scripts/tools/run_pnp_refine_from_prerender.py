import os
import argparse
import numpy as np
import torch

from pose_estimation.file_utils import get_checkpoint_arguments
from scene import load_data
from pose_estimation.refine_backends import prerender_adapter
from pose_estimation.optimization import optimize_camera_pose
from pose_estimation.error_computation import compute_translation_error, compute_angular_error
from utils.graphics_utils import fov2focal


def qvec2rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def load_w2c_txt(path: str):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            q = [float(x) for x in parts[1:5]]
            t = [float(x) for x in parts[5:8]]
            R = qvec2rotmat(q)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = np.array(t, dtype=np.float64)
            m[os.path.splitext(name)[0]] = w2c
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="i2dgs experiment dir containing cfg_args (e.g., ./output/garden)")
    ap.add_argument("--data_root", required=False, default="", help="unused (kept for clarity)")
    ap.add_argument("--init_w2c", required=True, help="init w2c txt (GS-CPR format)")
    ap.add_argument("--render_dir", required=True, help="pre-render directory containing per-camera *.png + *.npy")
    ap.add_argument("--device", default="cuda", help="cuda/cpu")
    ap.add_argument("--matcher", default="akaze_gms", choices=["akaze_gms", "sift", "lightglue", "loftr"])
    args = ap.parse_args()

    ckpt_args = get_checkpoint_arguments(args.exp_dir)
    scene_info = load_data(ckpt_args)
    cams = scene_info.test_cameras

    w2c_map = load_w2c_txt(args.init_w2c)
    backend = prerender_adapter
    backend_model = backend.load_model("", device=args.device, render_dir=args.render_dir)

    t_errs_init, a_errs_init = [], []
    t_errs_ref, a_errs_ref = [], []
    missing = 0

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for i, cam in enumerate(cams):
        name = getattr(cam, "image_name", None) or f"{i:05d}.png"
        key = os.path.splitext(name)[0]
        if key not in w2c_map:
            missing += 1
            continue

        # GT pose from cam
        w2c_gt = torch.eye(4, dtype=torch.float32, device=dev)
        w2c_gt[:3, :3] = torch.from_numpy(cam.R).T.to(dev)
        w2c_gt[:3, 3] = torch.from_numpy(cam.T).to(dev)
        c2w_gt = torch.inverse(w2c_gt)

        # init pose
        w2c_init_np = w2c_map[key]
        w2c_init = torch.from_numpy(w2c_init_np).to(dev, dtype=torch.float32)
        c2w_init = torch.inverse(w2c_init)

        # compute init errors
        gt_pos = torch.tensor([0., 0., 0., 1.], device=dev).unsqueeze(0) @ c2w_gt[:3, :].T
        pd_pos0 = torch.tensor([0., 0., 0., 1.], device=dev).unsqueeze(0) @ c2w_init[:3, :].T
        t_err0 = compute_translation_error(gt_pos, pd_pos0).item()
        a_err0 = compute_angular_error(c2w_gt[:3, :3], c2w_init[:3, :3]).item()
        t_errs_init.append(t_err0)
        a_errs_init.append(a_err0)

        # K for query
        fx = fov2focal(cam.FovX, cam.width)
        fy = fov2focal(cam.FovY, cam.height)
        K_query = np.array([[fx, 0.0, cam.width / 2.0], [0.0, fy, cam.height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        # refine using pre-rendered backend (no model rendering in refine loop)
        R_render = w2c_init_np[:3, :3]
        t_render = w2c_init_np[:3, 3]

        R_ref, t_ref, _ = optimize_camera_pose(
            camera_info=cam,
            model=backend_model,
            device=dev,
            K_query=K_query,
            R_render=R_render,
            t_render=t_render,
            matcher=args.matcher,
            refine_renderer="custom",
            refine_backend_adapter=backend,
            refine_backend_kwargs={"render_dir": args.render_dir},
        )

        w2c_ref = torch.eye(4, dtype=torch.float32, device=dev)
        w2c_ref[:3, :3] = torch.from_numpy(R_ref).to(dev, dtype=torch.float32)
        w2c_ref[:3, 3] = torch.from_numpy(t_ref).to(dev, dtype=torch.float32)
        c2w_ref = torch.inverse(w2c_ref)

        pd_pos1 = torch.tensor([0., 0., 0., 1.], device=dev).unsqueeze(0) @ c2w_ref[:3, :].T
        t_err1 = compute_translation_error(gt_pos, pd_pos1).item()
        a_err1 = compute_angular_error(c2w_gt[:3, :3], c2w_ref[:3, :3]).item()
        t_errs_ref.append(t_err1)
        a_errs_ref.append(a_err1)

    def _mean(x):
        return float(np.mean(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")

    print(f"Missing init poses for {missing}/{len(cams)} cameras.")
    print(f"INIT  mean trans err: {_mean(t_errs_init):.6f} | mean ang err: {_mean(a_errs_init):.6f}")
    print(f"REF   mean trans err: {_mean(t_errs_ref):.6f} | mean ang err: {_mean(a_errs_ref):.6f}")


if __name__ == "__main__":
    main()

