import math
import torch


# Try to import scatter_max from torch_scatter for efficient block selection
try:
    from torch_scatter import scatter_max
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalize tensor along specified dimension with zero-vector protection"""
    n = x.norm(dim=dim, keepdim=True)
    n_safe = n.clamp_min(eps)
    y = x / n_safe
    return torch.where(n > eps, y, torch.zeros_like(y))

def replace_if_zero(vec: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Replace zero vectors with fallback vectors"""
    need = (vec.norm(dim=-1, keepdim=True) < eps)
    return torch.where(need, fallback, vec)

def hemisphere_dirs(num_dirs: int, device: torch.device, method: str = "sobol", dtype=torch.float32):
    """Generate uniformly distributed directions on a hemisphere"""
    if method == "sobol":
        # Use Sobol sequence for quasi-random sampling
        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(dimension=2, scramble=True)
        uv = sobol.draw(num_dirs).to(device=device, dtype=dtype)
        u, v = uv.unbind(-1)
        theta = torch.acos(torch.clamp(1 - u, 0.0, 1.0))
        phi = 2 * math.pi * v
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        dirs = torch.stack((x, y, z), dim=1)
    else:
        # Use Fibonacci spiral sampling
        i = torch.arange(num_dirs, dtype=dtype, device=device)
        phi_g = (1.0 + math.sqrt(5.0)) / 2.0
        phi = 2.0 * math.pi * i / phi_g
        cos_t = torch.clamp(1.0 - (i + 0.5) / max(num_dirs, 1), 0.0, 1.0)
        sin_t = torch.sqrt(torch.clamp(1.0 - cos_t * cos_t, 0.0, 1.0))
        dirs = torch.stack((sin_t * torch.cos(phi), sin_t * torch.sin(phi), cos_t), dim=1)
    return safe_normalize(dirs, dim=-1)

def auto_block_select_with_mask(ray_wt: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor,
                                target: int, min_wt: float = 0.0, mask: torch.Tensor = None) -> torch.Tensor:
    """Select rays by dividing space into blocks and picking best ray from each block"""
    device = ray_wt.device
    K = int(ray_wt.numel())

    # Handle edge cases
    if K == 0 or target <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            nonzero_result = mask.nonzero(as_tuple=False)
            if nonzero_result.numel() > 0:
                valid_indices = nonzero_result.squeeze(1)
            else:
                valid_indices = torch.tensor([], device=device, dtype=torch.long)
        else:
            valid_indices = torch.arange(K, device=device)

        if valid_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Return all valid rays if fewer than target
        if valid_indices.numel() <= target:
            if min_wt > 0:
                valid_weights = ray_wt[valid_indices]
                weight_mask = valid_weights >= min_wt
                final_indices = valid_indices[weight_mask]
            else:
                final_indices = valid_indices
            return final_indices

        # Extract masked data
        mask_xs = xs[valid_indices]
        mask_ys = ys[valid_indices]
        mask_weights = ray_wt[valid_indices]
        K_masked = valid_indices.numel()
    else:
        # Use all data if no mask
        valid_indices = torch.arange(K, device=device)
        mask_xs = xs
        mask_ys = ys
        mask_weights = ray_wt
        K_masked = K

    # Return filtered results if already below target
    if K_masked <= target:
        if min_wt > 0:
            weight_mask = mask_weights >= min_wt
            if weight_mask.any():
                final_idx = valid_indices[weight_mask]
            else:
                final_idx = torch.tensor([], device=device, dtype=torch.long)
        else:
            final_idx = valid_indices
        return final_idx

    try:
        # Compute bounding box on CPU
        mask_xs_cpu = mask_xs.cpu()
        mask_ys_cpu = mask_ys.cpu()

        x_min = mask_xs_cpu.min().item()
        x_max = mask_xs_cpu.max().item()
        y_min = mask_ys_cpu.min().item()
        y_max = mask_ys_cpu.max().item()

        w_box = int(x_max - x_min + 1)
        h_box = int(y_max - y_min + 1)

        if w_box <= 0 or h_box <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Calculate grid dimensions based on aspect ratio
        ratio = w_box / max(h_box, 1)
        nx = max(int(math.sqrt(target * ratio)), 1)
        ny = max(int(math.ceil(target / nx)), 1)
        nx = min(nx, w_box)
        ny = min(ny, h_box)

        # Calculate block sizes
        bsx = max(int(math.ceil(w_box / nx)), 1)
        bsy = max(int(math.ceil(h_box / ny)), 1)

        # Assign points to blocks
        bx = ((mask_xs - x_min) // bsx).clamp(0, nx - 1)
        by = ((mask_ys - y_min) // bsy).clamp(0, ny - 1)
        block_id = (by * nx + bx).to(device=device, dtype=torch.long)

        num_blocks = nx * ny

        # Select best ray from each block
        if _HAS_SCATTER:
            best_w, best_pos = scatter_max(mask_weights, block_id, dim=0, dim_size=num_blocks)
            valid = (best_pos >= 0)
            if min_wt > 0:
                valid = valid & (best_w >= min_wt)
            sel_block = valid_indices[best_pos[valid]]
        else:
            # Fallback implementation without scatter_max
            sel_idx = []
            for b in range(num_blocks):
                block_mask = (block_id == b)
                if block_mask.any():
                    w, idx_local = torch.max(mask_weights[block_mask], dim=0)
                    if (min_wt <= 0) or (w >= min_wt):
                        block_nonzero = block_mask.nonzero(as_tuple=False)
                        if block_nonzero.numel() > 0:
                            pos = valid_indices[block_nonzero.squeeze(1)[idx_local]]
                            sel_idx.append(pos)
            sel_block = torch.stack(sel_idx) if sel_idx else torch.empty(0, dtype=torch.long, device=device)

        # Post-process selected blocks
        sel_block = torch.unique(sel_block, sorted=False)
        sel_block = sel_block[(sel_block >= 0) & (sel_block < len(ray_wt))]

        # Select top rays by weight if too many
        if sel_block.numel() > target:
            tmp_w = ray_wt[sel_block]
            top_local = torch.topk(tmp_w, target, largest=True).indices
            sel_block = sel_block[top_local]

        return sel_block

    except Exception as e:
        return torch.empty(0, dtype=torch.long, device=device)

def _to_hw3_normals(depth_normals: torch.Tensor, H: int, W: int, device):
    """Convert depth normals to (H, W, 3) format with safe normalization"""
    if depth_normals.dim() == 3:
        if depth_normals.shape[0] == 3:
            n = depth_normals.permute(1, 2, 0)
        elif depth_normals.shape[-1] == 3:
            n = depth_normals
        else:
            raise ValueError(f"Unsupported depth_normals shape: {tuple(depth_normals.shape)}")
    elif depth_normals.dim() == 2 and depth_normals.shape[-1] == 3 and depth_normals.shape[0] == H * W:
        n = depth_normals.view(H, W, 3)
    else:
        raise ValueError(f"Unsupported depth_normals shape: {tuple(depth_normals.shape)}")
    n = n.to(device).float()
    return safe_normalize(n, dim=-1)

def _make_K_from_fov(H: int, W: int, fovx: float, fovy: float, device, dtype):
    """Create camera intrinsic matrix from field of view"""
    fx = 0.5 * W / math.tan(0.5 * float(fovx))
    fy = 0.5 * H / math.tan(0.5 * float(fovy))
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], device=device, dtype=dtype)

def _unproject_xy_depth_to_world(xs: torch.Tensor, ys: torch.Tensor, depth_map: torch.Tensor, K: torch.Tensor,
                                 c2w: torch.Tensor):
    """Unproject 2D pixel coordinates with depth to 3D world coordinates"""
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
    """Compute ray direction from surface points to camera origin"""
    camera_origin = c2w[:3, 3]
    ray_dir = camera_origin.unsqueeze(0) - ray_ori
    return safe_normalize(ray_dir, dim=-1)

@torch.no_grad()
def sample_rays(
        ray_wt: torch.Tensor,
        H: int, W: int,
        img: torch.Tensor,
        depth_normals: torch.Tensor = None,
        mask: torch.BoolTensor = None,
        depth_map: torch.Tensor = None,
        K: torch.Tensor = None,
        c2w: torch.Tensor = None,
        fovx: float = None,
        fovy: float = None,
        target_rays: int = 100,
        num_dirs: int = 128,
        hemi_method: str = "sobol",
        min_wt_for_pick: float = 0.0,
        drop_invalid: bool = True,
):
    """Main function for sampling rays with proper mask handling"""
    # Input validation
    assert H is not None and W is not None, "H/W required"
    assert img is not None, "img required"
    assert depth_map is not None, "depth_map required"
    assert K is not None or (fovx is not None and fovy is not None), "K or fovx/fovy required"
    assert c2w is not None, "c2w required"
    assert mask is not None, "Mask is required"

    device = ray_wt.device
    dtype = ray_wt.dtype

    # Process input image to (H, W, 3) format
    if img.dim() == 3 and img.shape[0] == 3 and img.shape[1] == H and img.shape[2] == W:
        img_hw3 = img.permute(1, 2, 0)
    elif img.dim() == 3 and img.shape[:2] == (H, W) and img.shape[2] == 3:
        img_hw3 = img
    else:
        raise ValueError(f"Unsupported img shape: {tuple(img.shape)}")

    img_hw3 = img_hw3.to(device)
    if img_hw3.dtype.is_floating_point:
        img_hw3 = img_hw3.clamp(0.0, 1.0).float()
    else:
        img_hw3 = (img_hw3.float() / 255.0).clamp(0.0, 1.0)

    # Process mask
    if mask.dim() == 1 and mask.numel() == H * W:
        mask_2d = mask.view(H, W)
    elif mask.dim() == 2 and mask.shape == (H, W):
        mask_2d = mask
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    mask_2d = mask_2d.to(device).bool()
    mask_3d = mask_2d.unsqueeze(-1)

    # Flatten ray weight tensor to 1D
    if ray_wt.dim() == 1 and ray_wt.numel() == H * W:
        ray_wt_flat = ray_wt
    elif ray_wt.dim() == 2 and ray_wt.shape == (H, W):
        ray_wt_flat = ray_wt.view(-1)
    elif ray_wt.dim() == 3 and ray_wt.shape[-2:] == (H, W):
        ray_wt_flat = ray_wt.view(-1)
    else:
        raise ValueError(f"Unsupported ray_wt shape: {tuple(ray_wt.shape)}")

    ray_wt_flat = ray_wt_flat.to(device).float()

    # Process normal map
    if depth_normals is not None:
        n_hw3 = _to_hw3_normals(depth_normals, H, W, device)
        normal_norms = n_hw3.norm(dim=-1)
        valid_normals_mask = (normal_norms > 0.1) & (normal_norms < 2.0)
    else:
        n_hw3 = torch.zeros(H, W, 3, device=device, dtype=dtype)
        n_hw3[..., 2] = 1.0
        valid_normals_mask = torch.ones((H, W), dtype=torch.bool, device=device)

    # Process depth map
    if depth_map.dim() == 3 and depth_map.shape[0] == 1:
        depth_hw = depth_map[0]
    elif depth_map.dim() == 3 and depth_map.shape[-1] == 1:
        depth_hw = depth_map[..., 0]
    elif depth_map.dim() == 2 and depth_map.shape == (H, W):
        depth_hw = depth_map
    else:
        raise ValueError(f"Unsupported depth_map shape: {tuple(depth_map.shape)}")

    depth_hw = depth_hw.to(device=device, dtype=dtype)

    # Validate depth values
    depth_valid = torch.isfinite(depth_hw) & (depth_hw > 0) & (depth_hw < 100.0)

    # Create combined validity mask
    keep_mask_2d = mask_2d & valid_normals_mask & depth_valid

    # Create camera matrix if not provided
    if K is None:
        K = _make_K_from_fov(H, W, fovx, fovy, device, dtype)
    else:
        K = K.to(device=device, dtype=dtype)

    c2w = c2w.to(device=device, dtype=dtype)

    # Create coordinate grids
    ys_all, xs_all = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.long),
        torch.arange(W, device=device, dtype=torch.long),
        indexing='ij'
    )
    xs_flat = xs_all.reshape(-1)
    ys_flat = ys_all.reshape(-1)

    # Unproject all pixels to 3D world space
    ray_ori_all = _unproject_xy_depth_to_world(xs_flat, ys_flat, depth_hw, K, c2w)
    ray_dir_all = _compute_ray_direction_from_surface(ray_ori_all, c2w)

    # Get indices of valid rays
    keep_mask_flat = keep_mask_2d.view(-1)
    nonzero_result = keep_mask_flat.nonzero(as_tuple=False)
    if nonzero_result.numel() > 0:
        keep_idx = nonzero_result.squeeze(1).to(device=device, dtype=torch.long)
    else:
        keep_idx = torch.tensor([], device=device, dtype=torch.long)

    if keep_idx.numel() == 0:
        empty = torch.empty(0, 3, device=device, dtype=dtype)
        return empty, empty, empty

    Knum = int(keep_idx.numel())
    xs_all = (keep_idx % W).to(dtype=torch.long)
    ys_all = (keep_idx // W).to(dtype=torch.long)

    # Select base rays using block selection
    sel_block = auto_block_select_with_mask(
        ray_wt_flat[keep_idx], xs_all, ys_all,
        target_rays, min_wt=min_wt_for_pick,
        mask=None
    )

    if sel_block.numel() == 0:
        empty = torch.empty(0, 3, device=device, dtype=dtype)
        return empty, empty, empty

    sel_block = sel_block.to(device=device, dtype=torch.long)
    sel_block = torch.unique(sel_block, sorted=False)
    sel_block = sel_block[(sel_block >= 0) & (sel_block < Knum)]

    if sel_block.numel() == 0:
        empty = torch.empty(0, 3, device=device, dtype=dtype)
        return empty, empty, empty

    # Get selected pixel coordinates
    pick_idx = keep_idx[sel_block]
    xs_sel = (pick_idx % W).long()
    ys_sel = (pick_idx // W).long()

    # Extract data for selected rays
    normals_raw = n_hw3[ys_sel, xs_sel, :].to(dtype=dtype)
    ray_ori_sel = ray_ori_all[pick_idx].to(dtype=dtype)
    ray_dir_sel = ray_dir_all[pick_idx].to(dtype=dtype)

    # Handle zero vectors
    inc_fallback = ray_dir_sel
    normals = replace_if_zero(normals_raw, inc_fallback)
    normals = safe_normalize(normals, dim=-1)
    z_axis = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand_as(normals)
    normals = replace_if_zero(normals, z_axis)
    normals = safe_normalize(normals, dim=-1)

    ray_dir_sel = replace_if_zero(ray_dir_sel, normals)
    ray_dir_sel = safe_normalize(ray_dir_sel, dim=-1)

    # Filter out invalid vectors
    valid_vec = (normals.norm(dim=-1) > 0) & (ray_dir_sel.norm(dim=-1) > 0)
    if drop_invalid:
        if valid_vec.any():
            normals = normals[valid_vec]
            ray_ori_sel = ray_ori_sel[valid_vec]
            ray_dir_sel = ray_dir_sel[valid_vec]
            xs_sel = xs_sel[valid_vec]
            ys_sel = ys_sel[valid_vec]
        else:
            empty = torch.empty(0, 3, device=device, dtype=dtype)
            return empty, empty, empty

    B = ray_ori_sel.shape[0]

    # Create tangent space basis for hemisphere sampling
    ref = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand_as(normals)
    pole = normals[:, 2].abs() > 0.999
    alt = torch.tensor([0., 1., 0.], device=device, dtype=dtype).expand_as(normals)
    ref_vec = torch.where(pole.unsqueeze(1), alt, ref)
    t1 = safe_normalize(torch.cross(normals, ref_vec, dim=1), dim=1)
    t2 = torch.cross(normals, t1, dim=1)
    t2 = safe_normalize(t2, dim=1)

    # Generate hemispherical directions
    hemi = hemisphere_dirs(num_dirs, device, method=hemi_method, dtype=dtype)
    x_b, y_b, z_b = hemi.unbind(-1)

    # Transform hemisphere directions to local tangent space
    dirs = (t1[:, None, :] * x_b[None, :, None] +
            t2[:, None, :] * y_b[None, :, None] +
            normals[:, None, :] * z_b[None, :, None])
    dirs = safe_normalize(dirs, dim=2)

    # Prepare sampled directions and origins
    sample_dirs = dirs.reshape(-1, 3)
    sample_oris = ray_ori_sel.unsqueeze(1).expand(-1, num_dirs, 3).reshape(-1, 3)

    # Combine base rays with sampled hemispherical rays
    ray_dirs = torch.cat([ray_dir_sel, sample_dirs], dim=0)
    ray_oris = torch.cat([ray_ori_sel, sample_oris], dim=0)

    # Prepare colors
    base_cols = img_hw3[ys_sel, xs_sel, :].to(dtype=dtype)
    sample_cols = base_cols.unsqueeze(1).expand(-1, num_dirs, -1).reshape(-1, 3)
    colors = torch.cat([base_cols, sample_cols], dim=0)

    # Group rays by base point
    grouped_oris = torch.cat([ray_ori_sel[:, None, :], sample_oris.view(B, num_dirs, 3)], dim=1).reshape(-1, 3)
    grouped_dirs = torch.cat([ray_dir_sel[:, None, :], sample_dirs.view(B, num_dirs, 3)], dim=1).reshape(-1, 3)
    grouped_cols = torch.cat([colors[:B][:, None, :], colors[B:].view(B, num_dirs, 3)], dim=1).reshape(-1, 3)

    return grouped_oris, grouped_dirs, grouped_cols