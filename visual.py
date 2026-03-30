import torch
import numpy as np
import cv2
import os
import json
from typing import Tuple, Optional, Dict, Any, List
import argparse
from pathlib import Path
from PIL import Image

# Add necessary imports and globals.
_HAS_2DGS = False
try:
    from arguments import PipelineParams
    from scene.cameras import Camera
    from gaussian_renderer import render

    _HAS_2DGS = True
except Exception:
    _HAS_2DGS = False

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_numpy(x):
    """Helper: convert a tensor to a numpy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def correct_coordinate_system(c2w: np.ndarray) -> np.ndarray:
    """
    Fix coordinate-system differences with a more robust correction.
    """
    # More robust coordinate-system correction method.
    correction = np.eye(4)
    # Flip only the Y axis (as required by 2DGS coordinate conventions).
    correction[1, 1] = -1

    corrected_c2w = c2w @ correction
    return corrected_c2w


def load_camera_pose_from_json(pose_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera pose from JSON-formatted pose information.
    """
    # Extract position and rotation.
    position = np.array(pose_info["position"])
    rotation = np.array(pose_info["rotation"])

    # Build c2w matrix (camera to world).
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position

    print("Original c2w matrix:")
    print(c2w)

    # Apply coordinate-system correction.
    c2w_corrected = correct_coordinate_system(c2w)

    print("Corrected c2w matrix:")
    print(c2w_corrected)

    # Convert to w2c matrix (world to camera).
    w2c = np.linalg.inv(c2w_corrected)

    # Build intrinsics matrix.
    fx = pose_info["fx"]
    fy = pose_info["fy"]
    width = pose_info["width"]
    height = pose_info["height"]

    # Correct intrinsics to account for the image center.
    cx = width / 2
    cy = height / 2

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    print(f"Loaded camera pose: {pose_info['img_name']}")
    print(f"Position: {position}")
    print(f"Image size: {width}x{height}")
    print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    return w2c, intrinsics, c2w_corrected


def render_2dgs_view(
        model: Any,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        image_size: Tuple[int, int],
        output_path: Optional[str] = None,
        device: str = "cuda",
        pipeline_params: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a visualization image for a specified view using the 2DGS model.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("Warning: CUDA is not available; using CPU for rendering")

    device = torch.device(device)

    # Extract rotation and translation.
    R = pose[:3, :3]
    t = pose[:3, 3]

    width, height = image_size

    # Try 2DGS-specific rendering.
    if _HAS_2DGS:
        try:
            if pipeline_params is None:
                parser = argparse.ArgumentParser(add_help=False)
                pipeline_params = PipelineParams(parser)

            # Compute FOV with a more accurate method.
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]

            # FOV computation (in radians).
            fov_x = 2 * np.arctan(width / (2 * fx))
            fov_y = 2 * np.arctan(height / (2 * fy))

            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)

            print(f"FOV: {fov_x_deg:.2f} deg x {fov_y_deg:.2f} deg")

            # Create a dummy image.
            dummy_image = torch.zeros((3, height, width), device=device)

            # Create camera object; use transposed rotation matrix.
            cam = Camera(
                colmap_id=0,
                R=R.T,  # Use transposed rotation matrix (2DGS expected format).
                T=t,
                FoVx=fov_x,
                FoVy=fov_y,
                image=dummy_image,
                gt_alpha_mask=None,
                image_name="render_view",
                uid=0,
                data_device=device,
            )

            # Set background color.
            bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)

            # Render.
            with torch.no_grad():
                pkg = render(cam, model, pipeline_params, bg)

            # Extract RGB and depth.
            rgb_tensor = pkg.get("render", pkg.get("rgb"))
            if rgb_tensor is not None:
                rgb_tensor = rgb_tensor.clamp(0.0, 1.0).cpu()
            else:
                raise RuntimeError("2DGS rendering did not return RGB image")

            depth_tensor = pkg.get("depth", pkg.get("surf_depth"))
            if depth_tensor is None:
                print("Warning: 2DGS rendering did not return depth; creating a default depth map")
                depth_tensor = torch.ones((height, width), dtype=torch.float32)

            # Convert to numpy array.
            rgb_image = (rgb_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            depth_image = depth_tensor.cpu().numpy().astype(np.float32)

            # Check output image size; resize if it does not match expectation.
            if rgb_image.shape[:2] != (height, width):
                print(f"Rendered image size: {rgb_image.shape[:2]}; expected: {(height, width)}")
                rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_LINEAR)
                depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_LINEAR)

            return rgb_image, depth_image

        except Exception as e:
            print(f"2DGS-specific rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    return None, None


def render_camera_pose_info(
        model: Any,
        pose_info: Dict,
        output_path: str,
        device: str = "cuda",
        pipeline_params: Optional[Any] = None,
        flip_vertical: bool = False
) -> bool:
    """
    Render an image using the provided `camera_pose_info`.
    """
    try:
        # Extract w2c pose and intrinsics from pose information.
        pose_w2c, intrinsics, render_c2w = load_camera_pose_from_json(pose_info)

        # Get image size.
        width = pose_info["width"]
        height = pose_info["height"]
        image_size = (width, height)

        print(f"Start rendering: {pose_info['img_name']}")
        print(f"Image size: {width}x{height}")
        print(f"Vertical flip: {flip_vertical}")

        # Render.
        rgb, depth = render_2dgs_view(
            model=model,
            pose=pose_w2c,
            intrinsics=intrinsics,
            image_size=image_size,
            output_path=None,
            device=device,
            pipeline_params=pipeline_params
        )

        if rgb is None:
            print(f"Rendering failed: {output_path}")
            return False

        # Check rendering results.
        print(f"Render result shape: {rgb.shape}")
        print(f"Render result dtype: {rgb.dtype}")
        print(f"Render value range: [{rgb.min()}, {rgb.max()}]")

        # Optionally apply a vertical flip.
        if flip_vertical:
            print("Applying vertical flip")
            rgb = np.flipud(rgb)  # Vertical flip.

        # Save the image (use PIL to avoid OpenCV BGR issues).
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            # Save RGB image using PIL.
            Image.fromarray(rgb).save(output_path)
            print(f"Image saved: {output_path}")

        print(f"Rendering succeeded: {output_path}")
        return True

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_all_camera_poses(
        model: Any,
        camera_pose_info_list: List[Dict],
        output_dir: str,
        device: str = "cuda",
        flip_vertical: bool = False
):
    """
    Render all images corresponding to the given `camera_pose_info` list.
    """
    print("=== Start rendering camera pose images ===")

    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Create pipeline params.
    parser = argparse.ArgumentParser(add_help=False)
    pipeline_params = PipelineParams(parser)

    success_count = 0
    total_count = len(camera_pose_info_list)

    for idx, pose_info in enumerate(camera_pose_info_list):
        print(f"\n--- Rendering image {idx + 1}/{total_count}: {pose_info['img_name']} ---")

        # Generate output path.
        output_filename = f"rendered_{pose_info['img_name']}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Render current pose.
        success = render_camera_pose_info(
            model=model,
            pose_info=pose_info,
            output_path=output_path,
            device=device,
            pipeline_params=pipeline_params,
            flip_vertical=flip_vertical
        )

        if success:
            success_count += 1
            print(f"Rendering succeeded: {output_path}")
        else:
            print(f"Rendering failed: {output_path}")

    print(f"\n=== Done. Success {success_count}/{total_count}. Outputs in: {output_dir} ===")
    return success_count


# Main rendering function: render only the `camera_pose_info` corresponding images.
def main_render_camera_poses_only():
    """
    Main function: render only the images corresponding to `camera_pose_info`.
    """
    # Given camera pose information.
    camera_pose_info = {
        "id": 48,
        "img_name": "_DSC8729",
        "width": 4978,
        "height": 3300,
        "position": [0.8216700131733573, 6.578851269216608,-4.9879000602614256],
        "rotation":  [[0.9896318665687808, -0.018331174804166676, -0.14245257702790043], [-0.090158587787675, 0.6927815805367856, -0.7154894204089151], [0.11180428310097475, 0.7209144538010402, 0.6839460158382566]],
        "fy": 4528.0745533801455,
        "fx": 4528.597703297461
    }

    # If there are multiple `camera_pose_info`, they can be placed in a list.
    camera_pose_info_list = [camera_pose_info]

    # Model path.
    # Update this path to your local 2DGS checkpoint point_cloud.ply.
    model_path = "./path/to/point_cloud.ply"

    try:
        # Load model.
        from scene import GaussianModel

        model = GaussianModel(3)
        model.load_ply(model_path)
        print("Model loaded successfully")

        # Render only the images corresponding to `camera_pose_info`.
        print("\n=== Start rendering camera pose images ===")
        output_dir = "camera_pose_renders"

        success_count = render_all_camera_poses(
            model=model,
            camera_pose_info_list=camera_pose_info_list,
            output_dir=output_dir,
            device="cuda",
            flip_vertical=True,  # Apply vertical flip if needed.
        )

        if success_count > 0:
            print(f"\nDone. Rendered {success_count} images.")
            print(f"Outputs saved to: {output_dir}")
        else:
            print(f"\nRendering failed; please check the model and camera parameters")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


# Usage example.
if __name__ == "__main__":
    main_render_camera_poses_only()