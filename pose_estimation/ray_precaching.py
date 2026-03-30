import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from arguments import PipelineParams
from scene.scene_structure import SceneInfo
from scene.cameras import Camera
import torchvision.transforms as T


def _render_with_2dgs_for_rays(model, camera, pipe_params, bg_color, device, target_size=None):
    """Render surface depth, normals, and alpha using 2D Gaussian Splatting"""
    try:
        from gaussian_renderer import render

        # Render the scene
        render_pkg = render(camera, model, pipe_params, bg_color)

        # Extract rendering outputs
        surf_depth = render_pkg.get('surf_depth', None)
        surf_normal = render_pkg.get('surf_normal', None)
        rend_alpha = render_pkg.get('rend_alpha', None)

        if surf_depth is None:
            raise ValueError("2DGS renderer did not return surf_depth")
        if surf_normal is None:
            raise ValueError("2DGS renderer did not return surf_normal")
        if rend_alpha is None:
            raise ValueError("2DGS renderer did not return rend_alpha")

        # Process depth map
        if surf_depth.dim() == 3 and surf_depth.shape[0] == 1:
            surf_depth = surf_depth.squeeze(0)
        elif surf_depth.dim() == 3 and surf_depth.shape[0] == 3:
            surf_depth = surf_depth[0]

        # Process surface normals
        if surf_normal.dim() == 3 and surf_normal.shape[0] == 3:
            surf_normal = surf_normal.permute(1, 2, 0).contiguous()
        elif surf_normal.dim() == 2:
            surf_normal = surf_normal.unsqueeze(-1).expand(-1, -1, 3)

        # Process alpha channel
        if rend_alpha.dim() == 3 and rend_alpha.shape[0] == 1:
            rend_alpha = rend_alpha.squeeze(0)
        elif rend_alpha.dim() == 3 and rend_alpha.shape[0] == 3:
            rend_alpha = rend_alpha.mean(dim=0)

        # Resize to target dimensions if needed
        expected_H, expected_W = target_size if target_size else (camera.image_height, camera.image_width)

        if surf_depth.shape != (expected_H, expected_W):
            surf_depth = F.interpolate(
                surf_depth.unsqueeze(0).unsqueeze(0),
                size=(expected_H, expected_W),
                mode='bilinear'
            ).squeeze()

        if surf_normal.shape != (expected_H, expected_W, 3):
            if surf_normal.dim() == 3:
                surf_normal = surf_normal.permute(2, 0, 1)
            surf_normal = F.interpolate(
                surf_normal.unsqueeze(0),
                size=(expected_H, expected_W),
                mode='bilinear'
            ).squeeze(0).permute(1, 2, 0)

        if rend_alpha.shape != (expected_H, expected_W):
            rend_alpha = F.interpolate(
                rend_alpha.unsqueeze(0).unsqueeze(0),
                size=(expected_H, expected_W),
                mode='bilinear'
            ).squeeze()

        # Validate and clean depth data
        depth_min, depth_max = surf_depth.min(), surf_depth.max()
        if depth_max <= 0:
            surf_depth = torch.ones_like(surf_depth) * 5.0
        else:
            valid_depth_mask = (surf_depth > 0) & (surf_depth < 100.0) & torch.isfinite(surf_depth)
            if not valid_depth_mask.all():
                median_depth = surf_depth[valid_depth_mask].median()
                surf_depth = torch.where(valid_depth_mask, surf_depth, median_depth)

        # Validate and normalize surface normals
        if torch.allclose(surf_normal, torch.zeros_like(surf_normal)):
            surf_normal = torch.zeros_like(surf_normal)
            surf_normal[..., 2] = 1.0
        else:
            normal_norms = torch.norm(surf_normal, dim=-1)
            valid_normal_mask = (normal_norms > 0.1) & (normal_norms < 2.0) & torch.isfinite(normal_norms)
            if not valid_normal_mask.all():
                surf_normal = F.normalize(surf_normal, p=2, dim=-1)

        # Clamp alpha values
        rend_alpha = torch.clamp(rend_alpha, 0.0, 1.0)

        # Ensure tensors are on correct device
        surf_depth = surf_depth.to(device)
        surf_normal = surf_normal.to(device)
        rend_alpha = rend_alpha.to(device)

        return {
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'rend_alpha': rend_alpha
        }

    except Exception as e:
        raise RuntimeError(f"Rendering failed: {str(e)}")


def white_bg_mask_chw(rgb_chw: torch.Tensor) -> torch.BoolTensor:
    """Create mask for non-white pixels in CHW format"""
    white_threshold = 0.95
    return ~((rgb_chw > white_threshold).all(dim=0))


def white_bg_mask_hwc(rgb_hwc: torch.Tensor) -> torch.BoolTensor:
    """Create mask for non-white pixels in HWC format"""
    white_threshold = 0.95
    return ~((rgb_hwc > white_threshold).all(dim=-1))


def _create_fallback_rays(H, W, img_hwc, depth_map, K, c2w, mask, device, num_rays=256):
    """Create fallback rays when primary ray generation fails"""
    try:
        img_hwc = img_hwc.to(device)
        depth_map = depth_map.to(device)
        K = K.to(device)
        c2w = c2w.to(device)

        flat_img = img_hwc.reshape(-1, 3)

        # Determine valid pixel indices
        if mask is not None:
            mask = mask.to(device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            valid_flat = mask.reshape(-1) if mask.dim() == 2 else mask
            if valid_flat.any():
                valid_idx = valid_flat.nonzero(as_tuple=True)[0]
            else:
                valid_idx = torch.tensor([], device=device, dtype=torch.long)
        else:
            valid_idx = torch.arange(H * W, device=device)

        if valid_idx.numel() == 0:
            valid_idx = torch.arange(H * W, device=device)

        # Sample random rays
        k = min(num_rays, valid_idx.numel())
        if k == 0:
            empty_rays = torch.empty(0, 3, device=device)
            return empty_rays, empty_rays, empty_rays

        perm = torch.randperm(valid_idx.numel(), device=device)
        sel_indices = perm[:k]
        sel = valid_idx[sel_indices]

        if depth_map.dim() == 3:
            depth_map_2d = depth_map.squeeze(0)
        else:
            depth_map_2d = depth_map

        xs = (sel % W).long()
        ys = (sel // W).long()

        xs = torch.clamp(xs, 0, W - 1)
        ys = torch.clamp(ys, 0, H - 1)

        # Generate ray origins, directions and colors
        rays_ori = _unproject_xy_depth_to_world(xs, ys, depth_map_2d, K, c2w)
        rays_dir = _compute_ray_direction_from_surface(rays_ori, c2w)
        rays_rgb = flat_img[sel]

        return rays_ori, rays_dir, rays_rgb

    except Exception as e:
        # Return default rays in case of error
        default_ori = torch.zeros(1, 3, device=device)
        default_dir = torch.tensor([[0., 0., 1.]], device=device)
        default_rgb = torch.zeros(1, 3, device=device)
        return default_ori, default_dir, default_rgb


def _check_cache_files_exist(frame_dir):
    """Check if all required ray cache files exist"""
    cache_files = [
        os.path.join(frame_dir, "ori.npy"),
        os.path.join(frame_dir, "dir.npy"),
        os.path.join(frame_dir, "color.npy")
    ]
    return all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in cache_files)


def _cache_rays(idx: int, save_rays_dir: str, mask: torch.BoolTensor, H: int, W: int,
                img_hwc: torch.Tensor, surf_normals: torch.Tensor, depth_map: torch.Tensor,
                K: torch.Tensor, c2w: torch.Tensor, ray_wt: torch.Tensor, device=None):
    """Cache generated rays to disk for later use"""
    frame_dir = os.path.join(save_rays_dir, f"{idx:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Skip if cache already exists
    if _check_cache_files_exist(frame_dir):
        return

    # Preprocess depth and normal maps
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)

    if surf_normals.dim() == 3 and surf_normals.shape[0] == 3:
        surf_normals = surf_normals.permute(1, 2, 0).contiguous()

    # Generate rays using sampling module or fallback
    try:
        from pose_estimation.ray_generation_module import sample_rays
        rays_ori, rays_dir, rays_rgb = sample_rays(
            ray_wt=ray_wt, H=H, W=W, img=img_hwc, depth_normals=surf_normals,
            mask=mask, depth_map=depth_map, K=K, c2w=c2w
        )

        if rays_ori.numel() == 0:
            rays_ori, rays_dir, rays_rgb = _create_fallback_rays(
                H, W, img_hwc, depth_map, K, c2w, mask, device
            )
    except Exception as e:
        rays_ori, rays_dir, rays_rgb = _create_fallback_rays(
            H, W, img_hwc, depth_map, K, c2w, mask, device
        )

    # Filter out non-finite values
    finite_mask = (torch.isfinite(rays_ori).all(dim=1) &
                   torch.isfinite(rays_dir).all(dim=1) &
                   torch.isfinite(rays_rgb).all(dim=1))

    if finite_mask.any():
        rays_ori = rays_ori[finite_mask]
        rays_dir = rays_dir[finite_mask]
        rays_rgb = rays_rgb[finite_mask]
    else:
        # Create default rays if no valid ones found
        rays_ori = torch.zeros(1, 3, device=device)
        rays_dir = torch.tensor([[0., 0., 1.]], device=device)
        rays_rgb = torch.zeros(1, 3, device=device)

    # Save rays to disk
    np.save(f"{frame_dir}/ori.npy", rays_ori.detach().cpu().numpy())
    np.save(f"{frame_dir}/dir.npy", rays_dir.detach().cpu().numpy())
    np.save(f"{frame_dir}/color.npy", rays_rgb.detach().cpu().numpy())


def _unproject_xy_depth_to_world(xs: torch.Tensor, ys: torch.Tensor, depth_map: torch.Tensor,
                                 K: torch.Tensor, c2w: torch.Tensor):
    """Unproject 2D pixel coordinates with depth to 3D world coordinates"""
    if depth_map.dim() > 2:
        depth_map = depth_map.squeeze()

    xs = torch.clamp(xs, 0, depth_map.shape[1] - 1)
    ys = torch.clamp(ys, 0, depth_map.shape[0] - 1)

    z = depth_map[ys, xs]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Xc = (xs.float() - cx) / fx * z
    Yc = (ys.float() - cy) / fy * z
    Zc = z
    ones = torch.ones_like(Zc)
    Pc = torch.stack([Xc, Yc, Zc, ones], dim=-1)
    return (Pc @ c2w.T)[..., :3]


def _compute_ray_direction_from_surface(ray_ori: torch.Tensor, c2w: torch.Tensor):
    """Compute ray direction from surface point to camera origin"""
    camera_origin = c2w[:3, 3]
    ray_dir = camera_origin.unsqueeze(0) - ray_ori
    return F.normalize(ray_dir, dim=-1)


def create_pipeline_params():
    """Create pipeline parameters for 2DGS rendering"""
    parser = argparse.ArgumentParser()
    pipeline = PipelineParams(parser)
    args = argparse.Namespace()
    args.convert_SHs_python = False
    args.compute_cov3D_python = False
    args.depth_ratio = 0.0
    args.debug = False
    return pipeline.extract(args)


def calculate_target_size(orig_w, orig_h, target_width=1600):
    """Calculate target size while maintaining aspect ratio"""
    if orig_w <= target_width:
        return orig_w, orig_h, 1.0

    scale_factor = target_width / orig_w
    target_w = target_width
    target_h = max(1, int(orig_h * scale_factor))
    return target_w, target_h, scale_factor


def preprocess_rays_for_training(scene_info: SceneInfo, model, pipe, bg_color, device,
                                 save_rays_dir: str = "", target_width: int = 1600):
    """Preprocess and cache rays from all training cameras for pose estimation"""
    print("Pre-caching all rays from training cameras...")
    print(f"Number of training cameras: {len(scene_info.train_cameras)}")

    os.makedirs(save_rays_dir, exist_ok=True)

    # Prepare background color tensor
    if not isinstance(bg_color, torch.Tensor):
        bg_color_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)
    else:
        bg_color_tensor = bg_color

    if bg_color_tensor.dim() == 1:
        bg_color_tensor = bg_color_tensor.unsqueeze(0)

    # Process each training camera
    train_cameras = scene_info.train_cameras
    for local_idx, cam_info in enumerate(train_cameras):
        orig_w, orig_h = cam_info.image.size

        # Calculate target resolution
        target_w, target_h, scale_factor = calculate_target_size(orig_w, orig_h, target_width)
        resolution = (target_w, target_h)

        # Load and preprocess image
        channels = cam_info.image.split()
        resized_rgb = torch.cat([PILtoTorch(im, resolution) for im in channels[:3]], dim=0)
        resized_rgb = resized_rgb.float() / 255.0
        H_res, W_res = resized_rgb.shape[1], resized_rgb.shape[2]

        # Create foreground mask
        if len(channels) == 4:
            alpha_nn = channels[3].resize(resolution, resample=Image.NEAREST)
            resized_a = PILtoTorch(alpha_nn).float() / 255.0
            mask2d = resized_a.squeeze(0) > 0.3
        else:
            mask2d = white_bg_mask_chw(resized_rgb)

        mask_2d_bool = mask2d.to(device)
        mask_2d_float = mask_2d_bool.float()  # Convert to float for arithmetic operations
        img_hwc = resized_rgb.permute(1, 2, 0).contiguous().to(device)

        # Compute camera transformation matrices
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = torch.transpose(torch.from_numpy(cam_info.R), -1, -2).to(device)
        w2c[:3, -1] = torch.from_numpy(cam_info.T).to(device)
        c2w = torch.inverse(w2c)

        # Compute camera intrinsics
        fx = fov2focal(cam_info.FovX, W_res)
        fy = fov2focal(cam_info.FovY, H_res)
        K = torch.tensor([[fx, 0.0, (W_res - 1) / 2.0],
                          [0.0, fy, (H_res - 1) / 2.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        # Create camera for rendering
        cam_r = Camera(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=T.ToTensor()(cam_info.image.resize(resolution)),
            gt_alpha_mask=None,
            image_name=cam_info.image_name,
            uid=cam_info.uid,
            data_device=device,
        )

        # Render surface properties using 2DGS
        render_pkg = _render_with_2dgs_for_rays(model, cam_r, pipe, bg_color_tensor, device, (H_res, W_res))
        surf_depth = render_pkg['surf_depth']
        surf_normals = render_pkg['surf_normal']
        rend_alpha = render_pkg['rend_alpha']

        # Process depth map
        if surf_depth.dim() == 2:
            surf_depth_2d = surf_depth
        else:
            surf_depth_2d = surf_depth.squeeze(0)

        # Combine image mask with depth validity mask
        depth_valid_mask = (surf_depth_2d > 0.001) & (surf_depth_2d < 100.0) & torch.isfinite(surf_depth_2d)
        combined_mask = mask_2d_bool & depth_valid_mask

        # Fallback to simpler mask if too few valid pixels
        if combined_mask.sum().item() < 100:
            depth_valid_mask_simple = (surf_depth_2d > 0) & torch.isfinite(surf_depth_2d)
            combined_mask = mask_2d_bool & depth_valid_mask_simple

        mask_2d_bool = combined_mask
        mask_2d_float = mask_2d_bool.float()  # Update float mask
        flat_mask = mask_2d_bool.view(-1)

        # Skip if no valid pixels
        if flat_mask.sum() == 0:
            continue

        # Prepare data for ray caching
        masked_img_hwc = img_hwc * mask_2d_float.unsqueeze(-1)

        # Prepare surface normals - FIXED: Use float mask instead of bool mask for arithmetic
        if surf_normals.dim() == 3 and surf_normals.shape[2] == 3:
            if surf_normals.shape[:2] != mask_2d_bool.shape:
                surf_normals = F.interpolate(
                    surf_normals.permute(2, 0, 1).unsqueeze(0),
                    size=mask_2d_bool.shape,
                    mode='bilinear'
                ).squeeze(0).permute(1, 2, 0)
            forward_normal = torch.tensor([0.0, 0.0, 1.0], device=device)
            # Use float mask for arithmetic operations
            mask_3d_float = mask_2d_float.unsqueeze(-1)
            masked_normals_hwc = surf_normals * mask_3d_float + (1 - mask_3d_float) * forward_normal
        else:
            masked_normals_hwc = torch.zeros(H_res, W_res, 3, device=device)
            masked_normals_hwc[..., 2] = 1.0

        masked_normals_hwc = F.normalize(masked_normals_hwc, p=2, dim=-1)

        # Prepare depth map
        background_depth = 10.0
        masked_depth = surf_depth_2d * mask_2d_float + (1 - mask_2d_float) * background_depth

        # Compute ray weights
        ray_wt = rend_alpha.squeeze(0) * mask_2d_float

        if masked_depth.dim() == 3:
            masked_depth = masked_depth.squeeze(0)

        # Cache rays for this camera
        _cache_rays(
            idx=local_idx,
            save_rays_dir=save_rays_dir,
            mask=flat_mask,
            H=H_res,
            W=W_res,
            img_hwc=masked_img_hwc,
            surf_normals=masked_normals_hwc,
            depth_map=masked_depth,
            K=K,
            c2w=c2w,
            ray_wt=ray_wt,
            device=device
        )

        # Clear GPU cache periodically
        if local_idx % 10 == 9:
            torch.cuda.empty_cache()

    print("Training camera ray caching completed.")

    # Load all cached rays
    all_ori, all_dirs, all_rgb = [], [], []
    for local_idx in range(len(train_cameras)):
        frame_dir = os.path.join(save_rays_dir, f"{local_idx:05d}")

        if not _check_cache_files_exist(frame_dir):
            continue

        try:
            ori_data = torch.from_numpy(np.load(os.path.join(frame_dir, "ori.npy"))).to(device=device,
                                                                                        dtype=torch.float32)
            dir_data = torch.from_numpy(np.load(os.path.join(frame_dir, "dir.npy"))).to(device=device,
                                                                                        dtype=torch.float32)
            rgb_data = torch.from_numpy(np.load(os.path.join(frame_dir, "color.npy"))).to(device=device,
                                                                                          dtype=torch.float32)

            if ori_data.numel() > 0 and dir_data.numel() > 0 and rgb_data.numel() > 0:
                all_ori.append(ori_data)
                all_dirs.append(dir_data)
                all_rgb.append(rgb_data)
        except Exception as e:
            continue

    if len(all_ori) == 0:
        raise RuntimeError("No valid ray cache files found for training cameras!")

    # Concatenate all rays
    all_rays_ori = torch.cat(all_ori, dim=0)
    all_rays_dirs = torch.cat(all_dirs, dim=0)
    all_rays_rgb = torch.cat(all_rgb, dim=0)

    print(f"Successfully loaded rays from {len(all_ori)} training cameras, total {all_rays_ori.shape[0]} rays")

    return all_rays_ori, all_rays_dirs, all_rays_rgb