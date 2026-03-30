from __future__ import annotations
import argparse
from typing import List, Tuple, Optional, Dict, Any

import cv2 as cv
import numpy as np
import torch

try:
    from PIL import Image

    _HAS_PIL_TORCHVISION = True
except Exception:
    _HAS_PIL_TORCHVISION = False

_HAS_2DGS = False
try:
    from arguments import PipelineParams
    from scene.cameras import Camera
    from gaussian_renderer import render

    _HAS_2DGS = True
except Exception:
    _HAS_2DGS = False

# Global caches to reduce per-image overhead during refinement.
# These do not change optimization behavior; they only avoid re-initialization.
_PIPE_PARAMS_2DGS = None
_BG_WHITE_3 = None

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HAS_LIGHTGLUE = False
_LG_EXTRACTOR = None
_LG_MATCHER = None
try:
    from pose_estimation.LightGlue.lightglue import SuperPoint, LightGlue
    from pose_estimation.LightGlue.lightglue.utils import rbd

    _HAS_LIGHTGLUE = True
except ImportError:
    pass

_HAS_LOFTR = False
_LOFTR_MATCHER = None
try:
    from kornia.feature import LoFTR

    _HAS_LOFTR = True
except ImportError:
    pass


def _to_gray_u8(img: np.ndarray, *, use_clahe: bool = True, clip_limit: float = 2.5) -> np.ndarray:
    """Convert image to grayscale uint8 with optional CLAHE enhancement"""
    arr = img
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv.cvtColor(arr[..., :3], cv.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr) if np.isfinite(arr).any() else 1.0) <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr / 257.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if use_clahe:
        try:
            clahe = cv.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
            arr = clahe.apply(arr)
        except Exception:
            pass
    return np.ascontiguousarray(arr)


def _as_uint8_rgb_np(img: Any) -> np.ndarray:
    """
    Convert various image inputs to uint8 RGB numpy array (H,W,3).
    Supports PIL.Image, torch.Tensor ([H,W,C] or [C,H,W]), and numpy arrays.
    """
    if isinstance(img, np.ndarray):
        arr = img
    elif _HAS_PIL_TORCHVISION and isinstance(img, Image.Image):
        arr = np.asarray(img)
    elif torch.is_tensor(img):
        t = img.detach()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):  # [C,H,W]
            t = t.permute(1, 2, 0)
        arr = t.cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim == 2:
        arr = cv.cvtColor(arr, cv.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unexpected image shape: {getattr(arr, 'shape', None)}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr) if np.isfinite(arr).any() else 1.0)
            if mx <= 1.0 + 1e-6:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

    return np.ascontiguousarray(arr[..., :3])


def _get_2dgs_pipe_and_bg(device: torch.device):
    global _PIPE_PARAMS_2DGS, _BG_WHITE_3
    if _PIPE_PARAMS_2DGS is None:
        parser = argparse.ArgumentParser(add_help=False)
        _PIPE_PARAMS_2DGS = PipelineParams(parser)
    if _BG_WHITE_3 is None or (torch.is_tensor(_BG_WHITE_3) and _BG_WHITE_3.device != device):
        _BG_WHITE_3 = torch.ones(3, dtype=torch.float32, device=device)
    return _PIPE_PARAMS_2DGS, _BG_WHITE_3


def _morph_clean(mask: np.ndarray, k: int = 3, dilate: int = 1, close: int = 1) -> np.ndarray:
    """Clean mask using morphological operations"""
    m = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    k3 = np.ones((k, k), np.uint8)
    if close > 0:  m = cv.morphologyEx(m, cv.MORPH_CLOSE, k3, iterations=close)
    if dilate > 0: m = cv.dilate(m, k3, iterations=dilate)
    return m > 0


def _color_fg_mask(rgb_uint8: np.ndarray, white_thresh: int = 245) -> np.ndarray:
    """Create foreground mask by excluding white background"""
    return ~((rgb_uint8[..., 0] >= white_thresh) &
             (rgb_uint8[..., 1] >= white_thresh) &
             (rgb_uint8[..., 2] >= white_thresh))


def build_fg_mask_from_np(img: np.ndarray, white_thresh: int = 245, alpha_thr: float = 0.3) -> np.ndarray:
    """Build foreground mask from numpy image using alpha channel or color thresholding"""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3]
        m = (a > int(alpha_thr * 255))
    else:
        m = _color_fg_mask(img[..., :3], white_thresh=white_thresh)
    return _morph_clean(m, k=3, dilate=1, close=1)


def _filter_matches_by_masks(kp1, kp2, matches, mask1, mask2):
    """Filter matches based on foreground masks"""
    if mask1 is None and mask2 is None:
        return matches

    H1, W1 = (mask1.shape if mask1 is not None else (0, 0))
    H2, W2 = (mask2.shape if mask2 is not None else (0, 0))
    out = []

    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt

        if mask1 is not None:
            ix1 = min(W1 - 1, max(0, int(round(x1))))
            iy1 = min(H1 - 1, max(0, int(round(y1))))
            if not mask1[iy1, ix1]: continue

        if mask2 is not None:
            ix2 = min(W2 - 1, max(0, int(round(x2))))
            iy2 = min(H2 - 1, max(0, int(round(y2))))
            if not mask2[iy2, ix2]: continue

        out.append(m)
    return out


def _spatial_nms_matches(kp, matches, width: int, height: int, cell: int = 32, per_cell: int = 3):
    """Apply spatial non-maximum suppression to matches"""
    if len(matches) == 0:
        return matches

    grid_w = max(1, width // cell)
    grid_h = max(1, height // cell)
    buckets: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}

    for idx, m in enumerate(matches):
        x, y = kp[m.queryIdx].pt
        gx = min(grid_w - 1, max(0, int(x) // cell))
        gy = min(grid_h - 1, max(0, int(y) // cell))
        score = 1.0 / (1e-6 + float(getattr(m, 'distance', 0.0))) if getattr(m, 'distance', 0.0) > 0 else 1e6
        buckets.setdefault((gx, gy), []).append((score, idx))

    kept = []
    for key, arr in buckets.items():
        arr.sort(key=lambda t: -t[0])
        kept.extend([matches[i] for _, i in arr[:per_cell]])
    return kept


def _depth_is_stable(dmap: np.ndarray, u: float, v: float, win: int = 5, rel_thr: float = 0.15) -> bool:
    """Check if depth is stable in local neighborhood (robust to outliers)"""
    H, W = dmap.shape
    x0 = int(np.floor(u)) - win // 2
    y0 = int(np.floor(v)) - win // 2
    x1 = x0 + win
    y1 = y0 + win

    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return False

    patch = dmap[y0:y1, x0:x1].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0)]

    if patch.size < 4:
        return False
    # Use median + MAD for robustness to extreme values.
    med = float(np.median(patch))
    if med <= 1e-6:
        return False
    mad = float(np.median(np.abs(patch - med))) + 1e-9
    # Approximate std: 1.4826 * MAD.
    est_std = 1.4826 * mad
    return (est_std / med) <= rel_thr


def _bilinear_depth(dmap: np.ndarray, u: float, v: float) -> float:
    """Get bilinearly interpolated depth value"""
    H, W = dmap.shape
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = x0 + 1, y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return np.nan

    dx, dy = u - x0, v - y0
    z00, z10 = dmap[y0, x0], dmap[y0, x1]
    z01, z11 = dmap[y1, x0], dmap[y1, x1]
    vals = np.array([z00, z10, z01, z11], dtype=np.float64)
    valid = np.isfinite(vals) & (vals > 0)
    if not valid.any():
        return np.nan
    # Weighted interpolation using only valid neighbors.
    weights = np.array([
        (1 - dx) * (1 - dy),
        dx * (1 - dy),
        (1 - dx) * dy,
        dx * dy,
    ], dtype=np.float64)
    weights = weights * valid.astype(np.float64)
    if weights.sum() <= 1e-8:
        return np.nan
    weights /= weights.sum()
    return float((weights * vals).sum())


def do_feature_matching_SIFT(im1, im2, ratio: float = 0.86,
                             mask1=None, mask2=None, retry_without_mask_if_few=True, few_kp_thresh=60,
                             max_matches: int = 2000):
    """Feature matching using SIFT algorithm"""
    g1 = _to_gray_u8(im1, use_clahe=True)
    g2 = _to_gray_u8(im2, use_clahe=True)
    # Reduce keypoints for speed; usually minimal impact on PnP accuracy.
    sift = cv.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.2)

    def _detect(gray, m):
        return sift.detectAndCompute(gray, (m.astype(np.uint8) * 255) if (m is not None) else None)

    kp1, des1 = _detect(g1, mask1)
    kp2, des2 = _detect(g2, mask2)

    if retry_without_mask_if_few and ((len(kp1) < few_kp_thresh) or (len(kp2) < few_kp_thresh)):
        kp1, des1 = _detect(g1, None)
        kp2, des2 = _detect(g2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return list(kp1 or []), list(kp2 or []), []

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    des1 /= (des1.sum(axis=1, keepdims=True) + 1e-12)
    des2 /= (des2.sum(axis=1, keepdims=True) + 1e-12)
    des1 = np.sqrt(des1)
    des2 = np.sqrt(des2)

    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=512))
    m12 = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in m12 if m.distance < ratio * n.distance]
    # Sort by distance and cap match count to keep PnP fast.
    good.sort(key=lambda m: m.distance)
    if max_matches is not None and len(good) > max_matches:
        good = good[:max_matches]
    good = _filter_matches_by_masks(kp1, kp2, good, mask1, mask2)
    return list(kp1), list(kp2), good


def do_feature_matching_AKAZE_GMS(im1, im2,
                                  mask1=None, mask2=None,
                                  ratio: float = 0.50,
                                  crosscheck: bool = True):
    """Feature matching using AKAZE algorithm with GMS filtering"""
    g1 = _to_gray_u8(im1, use_clahe=True)
    g2 = _to_gray_u8(im2, use_clahe=True)
    akaze = cv.AKAZE_create()

    def _detect(gray, m):
        return akaze.detectAndCompute(gray, (m.astype(np.uint8) * 255) if (m is not None) else None)

    kp1, des1 = _detect(g1, mask1)
    kp2, des2 = _detect(g2, mask2)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return list(kp1 or []), list(kp2 or []), []

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    m12 = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in m12 if m.distance < ratio * n.distance]

    if crosscheck:
        m21 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False).knnMatch(des2, des1, k=2)
        good21 = [m for m, n in m21 if m.distance < ratio * n.distance]
        pair_12 = {(m.queryIdx, m.trainIdx) for m in good}
        pair_21 = {(m.trainIdx, m.queryIdx) for m in good21}
        sym = pair_12 & pair_21
        good = [cv.DMatch(i, j, 0) for (i, j) in sym]

    try:
        H1, W1 = g1.shape
        H2, W2 = g2.shape
        if hasattr(cv, "xfeatures2d") and hasattr(cv.xfeatures2d, "matchGMS") and len(good) > 0:
            gms = cv.xfeatures2d.matchGMS((W1, H1), (W2, H2), kp1, kp2, good,
                                          withRotation=True, withScale=True, thresholdFactor=6.0)
            good = list(gms)
    except Exception:
        pass

    good = _filter_matches_by_masks(kp1, kp2, good, mask1, mask2)
    return list(kp1), list(kp2), good


def do_feature_matching_LightGlue(im1, im2, feature_type='superpoint',
                                  device=_DEVICE, match_threshold=0.5,
                                  mask1=None, mask2=None):
    """Feature matching using LightGlue with SuperPoint features"""
    if not _HAS_LIGHTGLUE:
        return do_feature_matching_SIFT(im1, im2, ratio=0.75)

    try:
        # Reuse LightGlue/SuperPoint models to avoid rebuild overhead.
        if feature_type != 'superpoint':
            return do_feature_matching_SIFT(im1, im2, ratio=0.75)

        global _LG_EXTRACTOR, _LG_MATCHER

        # Rebuild on first use or when the device changes.
        if _LG_EXTRACTOR is None or next(_LG_EXTRACTOR.parameters()).device != torch.device(device):
            _LG_EXTRACTOR = SuperPoint(max_num_keypoints=2048).eval().to(device)
        if _LG_MATCHER is None or next(_LG_MATCHER.parameters()).device != torch.device(device):
            _LG_MATCHER = LightGlue(features=feature_type).eval().to(device)

        extractor = _LG_EXTRACTOR
        matcher = _LG_MATCHER

        def _preprocess_image(img):
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[..., :3]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
            else:
                img_tensor = img
            return img_tensor

        im1_tensor = _preprocess_image(im1)
        im2_tensor = _preprocess_image(im2)

        with torch.no_grad():
            feats0 = extractor.extract(im1_tensor)
            feats1 = extractor.extract(im2_tensor)
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0 = feats0['keypoints'].cpu().numpy()
        kpts1 = feats1['keypoints'].cpu().numpy()
        matches = matches01['matches'].cpu().numpy()
        scores = matches01['scores'].cpu().numpy()

        kp1 = [cv.KeyPoint(x=kp[0], y=kp[1], size=10) for kp in kpts0]
        kp2 = [cv.KeyPoint(x=kp[0], y=kp[1], size=10) for kp in kpts1]

        good_matches = []
        for (idx0, idx1), score in zip(matches, scores):
            if idx0 != -1 and idx1 != -1 and score > match_threshold:
                good_matches.append(cv.DMatch(int(idx0), int(idx1), 1.0 - score))

        good_matches = _filter_matches_by_masks(kp1, kp2, good_matches, mask1, mask2)
        return kp1, kp2, good_matches

    except Exception as e:
        return do_feature_matching_SIFT(im1, im2, ratio=0.75)


def do_feature_matching_LoFTR(im1, im2,
                              device=_DEVICE,
                              mask1=None, mask2=None):
    """Feature matching using LoFTR (dense matcher)"""
    if not _HAS_LOFTR:
        # Fall back to the default local feature matcher.
        return do_feature_matching_AKAZE_GMS(im1, im2, mask1=mask1, mask2=mask2)

    global _LOFTR_MATCHER

    try:
        if _LOFTR_MATCHER is None or next(_LOFTR_MATCHER.parameters()).device != torch.device(device):
            # Recommended pretrained weight names: "outdoor" / "indoor".
            _LOFTR_MATCHER = LoFTR(pretrained='outdoor').eval().to(device)

        # Convert to grayscale tensor in [0,1] with shape [1,1,H,W].
        def _prep(img, m):
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    g = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                elif img.ndim == 3 and img.shape[2] == 4:
                    g = cv.cvtColor(img[..., :3], cv.COLOR_RGB2GRAY)
                else:
                    g = img
                tens = torch.from_numpy(g).float() / 255.0
            else:
                # Allow passing tensors directly; convert to grayscale when needed.
                tens = img.float()
                if tens.ndim == 3:
                    # [C,H,W]
                    if tens.shape[0] == 3:
                        tens = 0.2989 * tens[0] + 0.5870 * tens[1] + 0.1140 * tens[2]
                    elif tens.shape[0] == 1:
                        tens = tens[0]
                elif tens.ndim == 2:
                    pass
            tens = tens.to(device)

            # Apply mask: zero out background to avoid useless regions.
            if m is not None:
                m_u8 = (m.astype(np.uint8) * 255) if m.dtype != np.uint8 else m
                m_t = torch.from_numpy(m_u8.astype(np.float32) / 255.0).to(device)
                tens = tens * m_t

            return tens[None, None, ...]  # [1,1,H,W]

        g1 = _prep(im1, mask1)
        g2 = _prep(im2, mask2)

        with torch.no_grad():
            out = _LOFTR_MATCHER({'image0': g1, 'image1': g2})

        if 'keypoints0' not in out or 'keypoints1' not in out:
            return [], [], []

        kpts0 = out['keypoints0'].cpu().numpy()
        kpts1 = out['keypoints1'].cpu().numpy()

        if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
            return [], [], []

        # LoFTR is one-to-one; build pairs by index.
        n = min(kpts0.shape[0], kpts1.shape[0])
        kp1 = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=8) for p in kpts0[:n]]
        kp2 = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=8) for p in kpts1[:n]]

        # Encode with unit distance; treated as uniform weights later.
        matches = [cv.DMatch(i, i, 0.0) for i in range(n)]

        # Filter by masks again to avoid boundary pixels outside the mask.
        matches = _filter_matches_by_masks(kp1, kp2, matches, mask1, mask2)
        return kp1, kp2, matches

    except Exception:
        # On any error, fall back to AKAZE+GMS for robustness.
        return do_feature_matching_AKAZE_GMS(im1, im2, mask1=mask1, mask2=mask2)


def _render_with_2dgs(model, camera, pipe_params, bg_color, device):
    """Render image and depth using 2D Gaussian Splatting"""
    try:
        bg = bg_color.to(device) if torch.is_tensor(bg_color) else torch.tensor(bg_color, device=device)

        with torch.no_grad():
            pkg = render(camera, model, pipe_params, bg)

        rgb_t = pkg.get("render", pkg.get("rgb")).clamp(0.0, 1.0).cpu()
        depth_t = pkg.get("surf_depth", None)

        if depth_t is None:
            raise RuntimeError("2DGS rendering failed to return depth")

        ref_img_np = (rgb_t.permute(1, 2, 0).contiguous().numpy() * 255.0).astype(np.uint8)
        ref_depth_np = depth_t.cpu().numpy().squeeze().astype(np.float32)

        H_r, W_r = ref_depth_np.shape
        fx_r = 0.5 * W_r / np.tan(camera.FoVx * 0.5)
        fy_r = 0.5 * H_r / np.tan(camera.FoVy * 0.5)
        cx_r, cy_r = W_r * 0.5, H_r * 0.5

        K_render_intr = np.array([[fx_r, 0.0, cx_r],
                                  [0.0, fy_r, cy_r],
                                  [0.0, 0.0, 1.0]], dtype=np.float32)

        return ref_img_np, ref_depth_np, K_render_intr

    except Exception as e:
        return None, None, None


def _backproject_depth_to_world(u: float, v: float, depth: float,
                                K: np.ndarray, Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    """Backproject 2D pixel with depth to 3D world coordinates"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Xc = np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float64)
    return Rcw @ Xc + tcw


def optimize_camera_pose(
        *,
        camera_info: Any,
        model: Any,
        device: torch.device,
        K_query: np.ndarray,
        R_render: np.ndarray,
        t_render: np.ndarray,
        matcher: str = "akaze_gms", # 'akaze_gms' |'lightglue'|  'sift'
        refine_renderer: str = "2dgs",  # '2dgs' | 'custom'
        refine_backend_adapter: Any = None,
        refine_backend_kwargs: Optional[Dict[str, Any]] = None,
        min_corresp: int = 30,
        ransac_reproj_err: float = 4,
        ransac_confidence: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Optimize camera pose using feature matching and PnP"""
    # Process query image (fast path: avoid PIL/torchvision conversions)
    q_np0 = _as_uint8_rgb_np(camera_info.image)
    oh, ow = q_np0.shape[:2]

    # Match the original scaling: longest side ~1600 px.
    scale = ow / 1600.0 if ow > 1600 else 1.0
    new_w, new_h = int(round(ow / scale)), int(round(oh / scale))
    if new_w != ow or new_h != oh:
        # Prefer Lanczos for quality; also typically faster than PIL in this pipeline.
        q_np = cv.resize(q_np0, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)
    else:
        q_np = q_np0

    # Scale the intrinsics consistently.
    sx, sy = new_w / float(ow), new_h / float(oh)
    K_query_scaled = K_query.copy()
    K_query_scaled[0, 0] *= sx
    K_query_scaled[1, 1] *= sy
    K_query_scaled[0, 2] *= sx
    K_query_scaled[1, 2] *= sy

    # Render reference view using selected backend (2DGS by default)
    ref_img_np, ref_depth_np, K_render_intr = None, None, None
    backend = (refine_renderer or "2dgs").lower()
    backend_kwargs = refine_backend_kwargs or {}

    if backend == "2dgs":
        if _HAS_2DGS:
            pipe_params, bg_color = _get_2dgs_pipe_and_bg(device)
            img_t = torch.from_numpy(q_np).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32) / 255.0
            cam_r = Camera(
                colmap_id=getattr(camera_info, "uid", 0),
                R=R_render.T, T=t_render,
                FoVx=camera_info.FovX,
                FoVy=getattr(camera_info, "FoVY", getattr(camera_info, "FovY")),
                image=img_t,
                gt_alpha_mask=None,
                image_name=getattr(camera_info, "image_name", "query"),
                uid=getattr(camera_info, "uid", 0),
                data_device=device,
            )
            ref_img_np, ref_depth_np, K_render_intr = _render_with_2dgs(model, cam_r, pipe_params, bg_color, device)
    else:
        # Custom backend: expected to provide a render_reference() function.
        # Signature:
        #   render_reference(model, camera_info, R_render, t_render, device, width, height, **kwargs)
        # Returns:
        #   ref_img_uint8(H,W,3), ref_depth_float32(H,W), K_render_intr(3,3) in pixels of ref_img.
        if refine_backend_adapter is not None and hasattr(refine_backend_adapter, "render_reference"):
            ref_img_np, ref_depth_np, K_render_intr = refine_backend_adapter.render_reference(
                model=model,
                camera_info=camera_info,
                R_render=R_render,
                t_render=t_render,
                device=device,
                width=int(new_w),
                height=int(new_h),
                **backend_kwargs,
            )

    if ref_img_np is None or ref_depth_np is None:
        return R_render, t_render, None

    # Create masks for feature matching
    fg = (np.isfinite(ref_depth_np) & (ref_depth_np > 0)).astype(np.uint8) * 255
    render_mask_depth = (fg.astype(np.float32) / 255.0) >= 0.5
    color_mask_render = build_fg_mask_from_np(ref_img_np, white_thresh=245, alpha_thr=0.3)
    render_mask = np.logical_or(render_mask_depth, color_mask_render)
    q_mask = build_fg_mask_from_np(q_np, white_thresh=245, alpha_thr=0.3)

    # Perform feature matching
    mname = (matcher or "akaze_gms").lower()

    if mname in ("lightglue", "superpoint", "disk"):
        feature_type = 'superpoint' if mname == 'lightglue' else mname
        kp_r, kp_q, matches = do_feature_matching_LightGlue(
            ref_img_np, q_np, feature_type=feature_type, device=device,
            match_threshold=0.8, mask1=render_mask, mask2=q_mask
        )
    elif mname in ("loftr",):
        kp_r, kp_q, matches = do_feature_matching_LoFTR(
            ref_img_np, q_np, device=device, mask1=render_mask, mask2=q_mask
        )
    elif mname in ("akaze", "gms", "akaze_gms"):
        kp_r, kp_q, matches = do_feature_matching_AKAZE_GMS(
            ref_img_np, q_np, mask1=render_mask, mask2=q_mask, ratio=0.85, crosscheck=True
        )
    elif mname in ("sift",):
        kp_r, kp_q, matches = do_feature_matching_SIFT(
            ref_img_np, q_np, ratio=0.75, mask1=render_mask, mask2=q_mask, max_matches=2500
        )
    else:
        kp_r, kp_q, matches = do_feature_matching_AKAZE_GMS(
            ref_img_np, q_np, mask1=render_mask, mask2=q_mask, ratio=0.85, crosscheck=True
        )

    # Filter and validate matches
    nR, nQ = len(kp_r), len(kp_q)
    matches = [m for m in matches if (0 <= m.queryIdx < nR) and (0 <= m.trainIdx < nQ)]
    if len(matches) < min_corresp:
        return R_render, t_render, None

    H_r, W_r = ref_depth_np.shape
    cell = max(24, max(W_r, H_r) // 80)
    per_cell = 6 if mname == "loftr" else (4 if mname in ("lightglue", "superpoint", "disk") else 3)
    matches = _spatial_nms_matches(kp_r, matches, width=W_r, height=H_r, cell=cell, per_cell=per_cell)
    max_used = 2500 if mname == "loftr" else 2000
    if len(matches) > max_used:
        matches = matches[:max_used]
        
    rel_thr_depth = 0.20 if mname == "loftr" else 0.15
    K_r = K_render_intr.astype(np.float64)
    Rr_c2w = R_render.astype(np.float64).T
    tr_c2w = -Rr_c2w @ t_render.astype(np.float64).reshape(3)

    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []

    for m in matches:
        u_r, v_r = kp_r[m.queryIdx].pt
        if not _depth_is_stable(ref_depth_np, u_r, v_r, win=5, rel_thr=rel_thr_depth):
            continue
        d = _bilinear_depth(ref_depth_np, u_r, v_r)
        if not np.isfinite(d) or d <= 0:
            continue
        Xw = _backproject_depth_to_world(u_r, v_r, d, K_r, Rr_c2w, tr_c2w)
        u_q, v_q = kp_q[m.trainIdx].pt
        obj_pts.append(Xw)
        img_pts.append([u_q, v_q])

    if len(obj_pts) < min_corresp:
        return R_render, t_render, None

    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

    # Solve PnP with RANSAC
    px_scale = max(new_w, new_h) / 1600.0
    ransac_err = float(ransac_reproj_err) * float(px_scale)

    Kq_scaled = K_query_scaled.astype(np.float64)
    distCoeffs = np.zeros(5, dtype=np.float64)

    n_corr = obj_pts.shape[0]
    if n_corr > 1500:
        iterations_count = 4000
    elif n_corr > 800:
        iterations_count = 5000
    elif n_corr > 400:
        iterations_count = 6500
    else:
        iterations_count = 8000

    # Two-stage RANSAC (speed, no accuracy loss):
    # - stage1: low iterations for easy frames
    # - stage2: fallback to full iterations only when stage1 is not confident
    def _pnp_ransac(iters: int):
        return cv.solvePnPRansac(
            objectPoints=obj_pts, imagePoints=img_pts,
            cameraMatrix=Kq_scaled, distCoeffs=distCoeffs,
            iterationsCount=int(iters), reprojectionError=ransac_err,
            confidence=float(max(ransac_confidence, 0.9995)),
            flags=cv.SOLVEPNP_EPNP
        )

    iters_full = int(iterations_count)
    # Conservative "fast" budget: never below 1200, never above 2500.
    iters_fast = int(max(1200, min(2500, iters_full // 3)))

    ok, rvec, tvec, inliers = _pnp_ransac(iters_fast)

    def _inlier_rmse(rv, tv, inl):
        if inl is None or inl.size < 6:
            return 1e9
        idx = inl.ravel()
        o_in = obj_pts[idx]
        x_in = img_pts[idx]
        proj, _ = cv.projectPoints(o_in, rv, tv, Kq_scaled, distCoeffs)
        err = np.linalg.norm(proj.reshape(-1, 2) - x_in, axis=1)
        return float(np.sqrt((err ** 2).mean())) if err.size > 0 else 1e9

    # Decide whether we trust stage1.
    # If inlier ratio is decent and RMSE is already small, skip stage2.
    n_inl = int(inliers.size) if (inliers is not None) else 0
    inl_ratio = float(n_inl) / float(max(1, obj_pts.shape[0]))
    rmse_stage1 = _inlier_rmse(rvec, tvec, inliers) if ok else 1e9
    trusted = ok and (n_inl >= max(18, min_corresp)) and (inl_ratio >= 0.15) and (rmse_stage1 <= (2.0 * px_scale))

    if not trusted:
        ok, rvec, tvec, inliers = _pnp_ransac(iters_full)

    if not ok:
        return R_render, t_render, None

    # Refine pose with iterative optimization
    # Key speed idea (no accuracy loss): if reprojection error is already very low,
    # skip extra LM refinement / re-weighted second solve.
    def _refine_and_metrics(o, x_all, i, rv, tv, scores, Kq_scaled, px_scale):
        if i is not None and i.size >= 6:
            o_in = o[i.ravel()]
            x_in = x_all[i.ravel()]
            s_in = scores[i.ravel()]

            ok2, rv, tv = cv.solvePnP(o_in, x_in, Kq_scaled, np.zeros(5, dtype=np.float64),
                                      rvec=rv, tvec=tv, useExtrinsicGuess=True,
                                      flags=cv.SOLVEPNP_ITERATIVE)
            if not ok2:
                return False, rv, tv, 0, 1e9

            # Compute reprojection error once.
            proj, _ = cv.projectPoints(o_in, rv, tv, Kq_scaled, np.zeros(5, dtype=np.float64))
            err = np.linalg.norm(proj.reshape(-1, 2) - x_in, axis=1)
            rmse0 = float(np.sqrt((err ** 2).mean())) if err.size > 0 else 1e9

            # If already extremely good, extra refinement is redundant.
            # Threshold is in pixels of the (possibly downscaled) query image.
            very_good_thr = 0.65 * px_scale
            if rmse0 <= very_good_thr or err.size < 32:
                return True, rv, tv, int(o_in.shape[0]), rmse0

            # Optional LM refinement: only when error is not yet tiny.
            try:
                cv.solvePnPRefineLM(o_in, x_in, Kq_scaled, np.zeros(5, dtype=np.float64), rv, tv)
            except Exception:
                pass

            # Recompute error after LM (still only once).
            proj_lm, _ = cv.projectPoints(o_in, rv, tv, Kq_scaled, np.zeros(5, dtype=np.float64))
            err_lm = np.linalg.norm(proj_lm.reshape(-1, 2) - x_in, axis=1)
            rmse_lm = float(np.sqrt((err_lm ** 2).mean())) if err_lm.size > 0 else rmse0

            # If LM already makes it very good, skip re-weighting pass.
            if rmse_lm <= (0.85 * px_scale):
                return True, rv, tv, int(o_in.shape[0]), rmse_lm

            # Re-weighted second solve on top correspondences (robust but can be expensive).
            sigma = 2.0 * px_scale
            w = (np.clip(s_in, 0.0, 1.0) ** 2.0) * np.exp(-(err_lm ** 2) / (2.0 * sigma * sigma))
            keep_idx = np.argsort(-w)
            k = max(6, int(0.8 * keep_idx.size))
            sel = keep_idx[:k]

            ok3, rv, tv = cv.solvePnP(o_in[sel], x_in[sel], Kq_scaled, np.zeros(5, dtype=np.float64),
                                      rvec=rv, tvec=tv, useExtrinsicGuess=True,
                                      flags=cv.SOLVEPNP_ITERATIVE)
            if not ok3:
                return True, rv, tv, int(o_in.shape[0]), rmse_lm

            proj2, _ = cv.projectPoints(o_in[sel], rv, tv, Kq_scaled, np.zeros(5, dtype=np.float64))
            err2 = np.linalg.norm(proj2.reshape(-1, 2) - x_in[sel], axis=1)
            rmse2 = float(np.sqrt((err2 ** 2).mean())) if err2.size > 0 else rmse_lm
            inlier_cnt = int(sel.size)
            return True, rv, tv, inlier_cnt, rmse2
        return False, rv, tv, 0, 1e9

    if mname in ("lightglue",):
        m_scores = np.array([1.0 - m.distance for m in matches], dtype=np.float64)
    elif mname in ("akaze", "gms", "akaze_gms", "loftr"):
        m_scores = np.ones(len(matches), dtype=np.float64)
    else:
        raw = np.array([max(0.0, 1.0 - float(m.distance)) for m in matches], dtype=np.float64)
        lo, hi = np.percentile(raw, [5.0, 95.0])
        denom = max(1e-6, hi - lo)
        m_scores = np.clip((raw - lo) / denom, 0.0, 1.0)

    ok, rvec, tvec, inlier_cnt, rmse = _refine_and_metrics(obj_pts, img_pts, inliers, rvec, tvec, m_scores, Kq_scaled,
                                                           px_scale)

    # Fallback to AP3P if initial refinement fails
    min_inliers_accept = max(10, min_corresp // 2) if mname == "loftr" else max(12, min_corresp // 2)
    rmse_threshold = 3.5 * px_scale if mname == "loftr" else 3.0 * px_scale
    if (inlier_cnt < min_inliers_accept) or (rmse > 1.8 * px_scale):
        ok2, rvec2, tvec2, inliers2 = cv.solvePnPRansac(
            objectPoints=obj_pts, imagePoints=img_pts,
            cameraMatrix=Kq_scaled, distCoeffs=np.zeros(5, dtype=np.float64),
            iterationsCount=5000, reprojectionError=1.5 * ransac_err,
            confidence=float(max(ransac_confidence, 0.9995)),
            flags=cv.SOLVEPNP_AP3P
        )
        if ok2:
            ok2, rvec2, tvec2, inlier_cnt2, rmse2 = _refine_and_metrics(obj_pts, img_pts, inliers2, rvec2, tvec2,
                                                                        m_scores, Kq_scaled, px_scale)
            if ok2 and (inlier_cnt2 > inlier_cnt or rmse2 < rmse):
                ok, rvec, tvec, inlier_cnt, rmse = ok2, rvec2, tvec2, inlier_cnt2, rmse2

    if inlier_cnt < min_inliers_accept or rmse > rmse_threshold:
        return R_render, t_render, None

    R_refined, _ = cv.Rodrigues(rvec)
    t_refined = tvec.reshape(3)

    return R_refined.astype(np.float64), t_refined.astype(np.float64), inliers


__all__ = [
    "do_feature_matching_SIFT",
    "do_feature_matching_AKAZE_GMS",
    "do_feature_matching_LightGlue",
    "do_feature_matching_LoFTR",
    "build_fg_mask_from_np",
    "optimize_camera_pose",
]

