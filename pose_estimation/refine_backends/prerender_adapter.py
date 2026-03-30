"""
Pre-render adapter for pose refinement (custom backend).

This adapter loads pre-rendered RGB (png) + depth (npy) pairs from `render_dir`
and exposes the minimal interface expected by `pose_estimation/optimization.py`:
  - load_model(model_path, device="cuda", render_dir=..., **kwargs)
  - render_reference(model, camera_info, R_render, t_render, device, width, height, **kwargs)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

__all__ = ["load_model", "render_reference"]


def _key_from_camera(camera_info: Any) -> str:
    name = getattr(camera_info, "image_name", None) or getattr(camera_info, "image", None)
    if name is None:
        # Fallback to uid/index-like identifiers if present.
        name = getattr(camera_info, "uid", None) or "0"
    base = os.path.splitext(str(name))[0]
    return base


def _compute_intrinsics_from_fov(camera_info: Any, width: int, height: int) -> np.ndarray:
    # i2dgs convention used elsewhere in this repo.
    fovx = float(getattr(camera_info, "FovX", getattr(camera_info, "fovx", 0.5)))
    fovy = float(getattr(camera_info, "FoVY", getattr(camera_info, "FoVy", getattr(camera_info, "fovy", fovx))))
    fx = 0.5 * float(width) / np.tan(0.5 * fovx)
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def load_model(model_path: str, device: str = "cuda", **kwargs) -> Dict[str, Any]:
    render_dir = kwargs.get("render_dir", "") or ""
    if not render_dir:
        raise ValueError("prerender_adapter.load_model: 'render_dir' is required")

    return {
        "render_dir": render_dir,
        "device": device,
        "_rgb_cache": {},   # key -> np.ndarray[uint8, H,W,3]
        "_depth_cache": {}, # key -> np.ndarray[float32, H,W]
    }


def _imread_rgb_uint8(path: str) -> Optional[np.ndarray]:
    if cv2 is not None:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if Image is not None:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                return np.asarray(im)
        except Exception:
            return None

    return None


def _imread_depth_float32(path: str) -> Optional[np.ndarray]:
    try:
        depth = np.load(path)
        if depth is None:
            return None
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth.squeeze()
        return depth.astype(np.float32, copy=False)
    except Exception:
        return None


def render_reference(
    *,
    model: Any,
    camera_info: Any,
    R_render: np.ndarray,
    t_render: np.ndarray,
    device: Any,
    width: int,
    height: int,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if model is None or not isinstance(model, dict):
        raise RuntimeError("prerender_adapter: invalid model bundle. Call load_model() first.")

    render_dir = model["render_dir"]
    key = _key_from_camera(camera_info)

    rgb_cache: Dict[str, np.ndarray] = model.setdefault("_rgb_cache", {})
    depth_cache: Dict[str, np.ndarray] = model.setdefault("_depth_cache", {})

    rgb_path = os.path.join(render_dir, f"{key}.png")
    depth_path = os.path.join(render_dir, f"{key}.npy")

    ref_img = rgb_cache.get(key)
    ref_depth = depth_cache.get(key)

    if ref_img is None:
        ref_img = _imread_rgb_uint8(rgb_path)
        if ref_img is None:
            return None, None, None
        rgb_cache[key] = ref_img

    if ref_depth is None:
        ref_depth = _imread_depth_float32(depth_path)
        if ref_depth is None:
            return None, None, None
        depth_cache[key] = ref_depth

    # Ensure sizes match the requested width/height (after query downscaling).
    if int(ref_img.shape[1]) != int(width) or int(ref_img.shape[0]) != int(height):
        if cv2 is not None:
            ref_img = cv2.resize(ref_img, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
            ref_depth = cv2.resize(ref_depth, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        else:
            # PIL path could be added, but keep it minimal: if no cv2, return unresized.
            pass

    K_render_intr = _compute_intrinsics_from_fov(camera_info, int(width), int(height))

    return ref_img.astype(np.uint8, copy=False), ref_depth.astype(np.float32, copy=False), K_render_intr

