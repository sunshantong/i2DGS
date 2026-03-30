import time
from typing import List, Optional
import glob
from tqdm import tqdm
import os
import io
import re
import contextlib
from statistics import mean
from pose_estimation.error_computation import (
    compute_translation_error,
    compute_angular_error,
)
import torch
import numpy as np
from pose_estimation.optimization import optimize_camera_pose as refine_pose
from pose_estimation.line_intersection import (
    compute_line_intersection_impl2,
    exclude_negatives,
    make_rotation_mat,
)
from scene.scene_structure import CameraInfo
from utils.graphics_utils import fov2focal


class _StdoutTensorFilter(io.TextIOBase):
    """Filter noisy debug tensor dumps from external refine backends."""

    _tensor_line = re.compile(r"^\s*tensor\(\[.*device='cuda:[0-9]+'\)\s*$")

    def __init__(self, downstream):
        self._downstream = downstream

    def write(self, s):
        if not s:
            return 0
        # Keep progress-bar fragments/newlines intact, only suppress full tensor debug lines.
        if self._tensor_line.match(s.strip()):
            return len(s)
        self._downstream.write(s)
        return len(s)

    def flush(self):
        self._downstream.flush()


def test_pose_estimation(
        model,
        cameras_info: List[CameraInfo],
        id_module,
        device,
        model_up,
        all_rays_ori,
        all_rays_dirs,
        all_rays_rgb,
        sequence_id="",
        category_id="",
        loss_fn=None,
        optimize=True,
        save_rays_dir="",
        refine_renderer: str = "2dgs",
        refine_matcher: str = "akaze_gms",
        refine_backend_adapter=None,
        refine_backend_kwargs: Optional[dict] = None,
        refine_model=None,
):
    """Test pose estimation pipeline with optional camera pose refinement"""
    id_module.eval()
    translation_errors = []
    angular_errors = []
    avg_loss_scores = []
    recalls = []
    results = []
    base_name = f"{sequence_id}_{category_id}"
    start_time = time.time()
    print(f"Total rays: {all_rays_ori.shape[0]}")

    module_device = next(id_module.parameters()).device

    frame_dirs = sorted(glob.glob(os.path.join(save_rays_dir, "[0-9]" * 5)))
    ray_counts_per_frame = []

    start_indices = [0]
    for fd in frame_dirs:
        rays_ori = torch.from_numpy(np.load(os.path.join(fd, "ori.npy")))
        count = rays_ori.shape[0]
        ray_counts_per_frame.append(count)
        start_indices.append(start_indices[-1] + count)

    num_frames = len(ray_counts_per_frame)
    print(f"Splitting {all_rays_ori.shape[0]} rays into {num_frames} frames")

    ray_feats_per_frame = []
    print("Computing frame features...")

    all_rays_ori = all_rays_ori.to(module_device, non_blocking=True)
    all_rays_dirs = all_rays_dirs.to(module_device, non_blocking=True)
    all_rays_rgb = all_rays_rgb.to(module_device, non_blocking=True)

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Computing rays features"):
            start_idx = start_indices[i]
            end_idx = start_indices[i + 1]

            frame_ori = all_rays_ori[start_idx:end_idx]
            frame_dirs = all_rays_dirs[start_idx:end_idx]
            frame_rgb = all_rays_rgb[start_idx:end_idx]

            frame_feat = id_module.ray_preprocessor(frame_ori, frame_dirs, frame_rgb)
            ray_feats_per_frame.append(frame_feat)

    print(f"Computed rays features of {num_frames} frames")

    for img_idx, camera_info in tqdm(enumerate(cameras_info)):
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = torch.from_numpy(camera_info.R).T.to(device)
        w2c[:3, 3] = torch.from_numpy(camera_info.T).to(device)
        c2w = torch.inverse(w2c)

        focalX = fov2focal(camera_info.FovX, camera_info.width)
        focalY = fov2focal(camera_info.FovY, camera_info.height)
        target_K = torch.tensor([
            [focalX, 0.0, camera_info.width / 2],
            [0.0, focalY, camera_info.height / 2],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)

        tensor_image = torch.from_numpy(np.array(camera_info.image))
        obs_img = tensor_image.to(module_device, torch.float32) / 255.0

        if obs_img.shape[-1] == 4:
            alpha = obs_img[..., -1]
            mask_img = alpha > 0.3
            obs_img = obs_img[..., :3] * alpha.unsqueeze(-1) + (1 - alpha).unsqueeze(-1)
        else:
            mask_img = ~((obs_img[..., :3] == 1.0).all(dim=-1))

        with torch.no_grad():
            img_feat_wpe, img_feat_flat, img_feat = id_module.backbone_wrapper(obs_img, mask_img)
            cam_up_token = id_module.camera_direction_prediction_network(img_feat)
            cam_up_aug = getattr(id_module, "camera_up_out_augmentation", None)
            if callable(cam_up_aug):
                cam_up_token = torch.nn.functional.normalize(cam_up_aug(cam_up_token), dim=-1)
            else:
                cam_up_token = torch.nn.functional.normalize(cam_up_token, dim=-1)

        cam_up_token = cam_up_token.squeeze()
        if cam_up_token.ndim != 1 or cam_up_token.shape[0] != 3:
            raise RuntimeError(f"Unexpected cam_up_token shape: {tuple(cam_up_token.shape)}")

        frame_id = f"{base_name}_{img_idx:05d}"

        all_scores = []
        all_top_scores = []
        all_top_indices = []
        current_offset = 0

        with torch.no_grad():
            for frame_idx, feat_rays in enumerate(ray_feats_per_frame):
                attn_map = id_module.attention(img_feat_wpe, feat_rays, mask=None)
                scores = attn_map.sum(dim=0)

                top_k_this_frame = min(25, scores.shape[0])
                topk = torch.topk(scores, k=top_k_this_frame, largest=True)
                local_idx = topk.indices
                local_scores = topk.values

                global_idx = local_idx + current_offset
                all_top_indices.append(global_idx)
                all_top_scores.append(local_scores)

                current_offset += scores.shape[0]
                all_scores.append(scores)

            all_top_scores = torch.cat(all_top_scores, dim=0)
            all_top_indices = torch.cat(all_top_indices, dim=0)
            pred_scores = torch.cat(all_scores, dim=0)

            top_k_global = all_top_scores.shape[0]
            topk = torch.topk(all_top_scores, k=top_k_global, largest=True)
            weights = topk.values
            idx = all_top_indices[topk.indices]
            weights = (weights - weights.min()).clamp(min=0) + 1e-8
            weights = weights / weights.sum()

        avg_score, recall = -1.0, -1.0
        if loss_fn is not None:
            avg_score, target_scores = loss_fn(
                pred_scores, c2w, target_K,
                all_rays_ori, all_rays_dirs,
                all_rays_ori.shape[0],
                id_module.backbone_wrapper.backbone_wh,
                model_up=cam_up_token
            )
            avg_score = avg_score.item()
            loc2 = torch.topk(target_scores, k=top_k_global).indices
            recall = float((torch.isin(idx, loc2)).sum() / top_k_global)

        avg_loss_scores.append(avg_score)
        recalls.append(recall)

        origins = all_rays_ori[idx]
        unique_o, inverse = torch.unique(origins, return_inverse=True, dim=0)
        M = unique_o.size(0)
        group_max = torch.full((M,), float('-inf'), device=weights.device)
        group_max.scatter_reduce_(0, inverse, weights, reduce="amax", include_self=True)
        mask = weights == group_max[inverse]
        idx = idx[mask]
        weights = weights[mask]
        if weights.sum() <= 0:
            raise ValueError("All ray weights are zero")
        weights = weights / weights.sum()

        center = compute_line_intersection_impl2(
            all_rays_ori[idx],
            all_rays_dirs[idx],
            weights=weights
        )
        weights = weights * exclude_negatives(center, all_rays_ori[idx], all_rays_dirs[idx])
        weights = (weights - weights.min()).clamp(min=0) + 1e-8
        weights = weights / weights.sum()
        center = compute_line_intersection_impl2(
            all_rays_ori[idx],
            all_rays_dirs[idx],
            weights=weights
        )
        if not torch.isfinite(center).all():
            center = (all_rays_ori[idx] * weights.unsqueeze(-1)).sum(dim=0)

        watch_dir = (all_rays_dirs[idx] * weights.unsqueeze(-1)).sum(dim=0)
        watch_dir = watch_dir / watch_dir.norm()

        R_mat = make_rotation_mat(-watch_dir, cam_up_token)
        if torch.linalg.det(R_mat) < 1e-7:
            R_mat = torch.eye(3, device=device)

        c2w_pred = torch.eye(4, device=device)
        c2w_pred[:3, :3] = torch.linalg.inv(R_mat)
        c2w_pred[:3, 3] = center

        if optimize:
            w2c_pred = torch.inverse(c2w_pred)
            R_render = w2c_pred[:3, :3].cpu().numpy()
            t_render = w2c_pred[:3, 3].cpu().numpy()
            K_tgt = target_K.cpu().numpy()

            try:
                # Some external backends print huge tensor debug lines every frame.
                # Filter those specific lines without muting normal warnings/errors.
                filtered_stdout = _StdoutTensorFilter(os.sys.stdout)
                with contextlib.redirect_stdout(filtered_stdout):
                    R_refined, t_refined, inliers = refine_pose(
                        camera_info=camera_info,
                        model=(refine_model if refine_model is not None else model),
                        device=device,
                        K_query=K_tgt,
                        R_render=R_render,
                        t_render=t_render,
                        matcher=refine_matcher,
                        refine_renderer=refine_renderer,
                        refine_backend_adapter=refine_backend_adapter,
                        refine_backend_kwargs=refine_backend_kwargs,
                    )
            except RuntimeError as e:
                if "Too few valid matches" in str(e):
                    print(f"[Warning] Frame {frame_id}: {e}, skipping refine.")
                    R_refined, t_refined, inliers = R_render, t_render, None
                else:
                    raise

            w2c_opt = torch.eye(4, device=device)
            w2c_opt[:3, :3] = torch.from_numpy(R_refined).to(device)
            w2c_opt[:3, 3] = torch.from_numpy(t_refined).to(device)
            c2w_pred = torch.inverse(w2c_opt)

        gt_pos = torch.tensor([0., 0., 0., 1.], device=c2w.device).unsqueeze(0) @ c2w[:3, :].T
        pd_pos = torch.tensor([0., 0., 0., 1.], device=c2w_pred.device).unsqueeze(0) @ c2w_pred[:3, :].T

        t_err = compute_translation_error(gt_pos, pd_pos)
        a_err = compute_angular_error(c2w[:3, :3], c2w_pred[:3, :3])

        translation_errors.append(t_err.item())
        angular_errors.append(a_err.item())

        results.append({
            "sequence_id": sequence_id,
            "category_name": category_id,
            "frame_id": img_idx,
            "loss": recall,
            "scores_loss": avg_score,
            "recall": recall,
            "pred_c2w": c2w_pred.cpu().tolist(),
            "gt_c2w": c2w.cpu().tolist(),
        })

    total_time = time.time() - start_time
    print(f"Avg loss: {mean(avg_loss_scores):.4f}, Avg recall: {mean(recalls):.4f}")
    print(f"Trans err: {mean(translation_errors):.4f}, Ang err: {mean(angular_errors):.4f}")
    print(f"Time per img: {total_time / len(cameras_info):.4f}s")


    return results, mean(translation_errors), mean(angular_errors), mean(avg_loss_scores), mean(recalls)

