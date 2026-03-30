import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import json
import traceback
import glob
import random
import numpy as np
import torch
from typing import Optional
import importlib
from pose_estimation.file_utils import parse_exp_dir, get_checkpoint_arguments, dotdict
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.opt import parse_args
from pose_estimation.test import test_pose_estimation
from scene import GaussianModel, load_data
from pose_estimation.distance_based_loss import DistanceBasedScoreLoss
from pose_estimation.ray_precaching import create_pipeline_params
from pose_estimation.train import train_id_module


def set_global_seed(seed: int = 0, deterministic: bool = True):
    """Set random seed for reproducibility"""
    try:
        seed = int(seed)
    except Exception:
        seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path: str, device: str, sh_degree: int = 3):
    """Load Gaussian model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model = GaussianModel(sh_degree)
    model.load_ply(checkpoint_path)
    model = model.to(device)
    return model


def get_rays_save_dir(checkpoint_args_filepath: str):
    """Create directory for saving ray cache"""
    cfg_args_dir = os.path.dirname(checkpoint_args_filepath)
    rays_save_dir = os.path.join(cfg_args_dir, "rays")
    os.makedirs(rays_save_dir, exist_ok=True)
    print(f"Rays will be saved to: {rays_save_dir}")
    return rays_save_dir


def pretrain_single_object(
        checkpoint_filepath: str,
        checkpoint_args: dotdict,
        exp_dir_filepath: str,
        object_id: str,
        category_name: str,
        starting_seed: int = None,
        lock_backbone: bool = True,
        device: str = "cuda",
        refine_renderer: str = "2dgs",
        refine_matcher: str = "akaze_gms",
        refine_backend_adapter=None,
        refine_backend_kwargs: "Optional[dict]" = None,
        refine_model_path: str = "",
):
    """Train identification module for a single object"""
    try:
        # Set random seed for reproducibility
        if starting_seed is not None:
            set_global_seed(starting_seed, deterministic=True)
            print(f"Using seed: {starting_seed} for object {object_id}")
        else:
            print(f"Running without fixed random seed for object {object_id}")

        print("data_path: ", checkpoint_args.source_path)

        cfg_args_path = os.path.join(exp_dir_filepath, "cfg_args")
        if not os.path.exists(cfg_args_path):
            cfg_args_path = exp_dir_filepath

        save_rays_dir = get_rays_save_dir(cfg_args_path)
        gs_model = load_model(checkpoint_filepath, device, sh_degree=checkpoint_args.sh_degree)

        # Load model used only for pose refinement stage (2DGS by default; can use a custom backend).
        refine_model = gs_model
        if (refine_renderer or "2dgs").lower() != "2dgs":
            if not refine_model_path:
                raise ValueError("--refine_model_path is required when --refine_renderer != 2dgs")
            if refine_backend_adapter is None or not hasattr(refine_backend_adapter, "load_model"):
                raise ValueError("refine backend adapter must provide load_model(model_path, device, **kwargs)")
            refine_model = refine_backend_adapter.load_model(
                refine_model_path, device=device, **(refine_backend_kwargs or {})
            )

        if checkpoint_args.fps_sampling is None:
            checkpoint_args.fps_sampling = -1

        # Load scene data and create pipeline
        scene_info = load_data(checkpoint_args)
        pipe = create_pipeline_params()
        bg_color = [1, 1, 1] if getattr(checkpoint_args, 'white_background', False) else [0, 0, 0]

        # Initialize identification module
        backbone_type = "dino"
        id_module = (
            IdentificationModule(backbone_type=backbone_type)
            .to(device, non_blocking=True)
            .train()
        )

        # Optionally freeze backbone parameters
        if lock_backbone:
            for parameter in id_module.backbone_wrapper.parameters():
                parameter.requires_grad = False

        # Check for existing checkpoint
        start_iterations = 0
        id_module_ckpt_path = os.path.join(exp_dir_filepath, "id_module.th")
        if os.path.exists(id_module_ckpt_path):
            print("Found best checkpoint, skipping training phase.")
            ckpt_dict = torch.load(id_module_ckpt_path, map_location=device)
            id_module.load_state_dict(ckpt_dict["model_state_dict"])
        else:
            # Train identification module
            train_id_module(
                id_module_ckpt_path, device, id_module,
                scene_info,
                object_id, category_name,
                gs_model,
                pipe, bg_color,
                start_iterations=start_iterations,
                lock_backbone=lock_backbone,
                save_rays_dir=save_rays_dir,
            )

        print("Training complete, starting testing phase...")
        print("Testing performances...")

        # Compute model up direction from training cameras
        model_up_np = np.mean(
            np.asarray(
                [train_camera.R[:3, 1] for train_camera in scene_info.train_cameras],
                dtype=np.float32,
            ),
            axis=0,
        )
        model_up = torch.from_numpy(model_up_np).to(device=device, non_blocking=True)

        # Load all cached rays
        frame_dirs = sorted(glob.glob(os.path.join(save_rays_dir, "[0-9]" * 5)))
        all_ori, all_dirs, all_rgb = [], [], []
        for fd in frame_dirs:
            all_ori.append(torch.from_numpy(np.load(os.path.join(fd, "ori.npy"))).to(device).to(torch.float32))
            all_dirs.append(torch.from_numpy(np.load(os.path.join(fd, "dir.npy"))).to(device).to(torch.float32))
            all_rgb.append(torch.from_numpy(np.load(os.path.join(fd, "color.npy"))).to(device).to(torch.float32))
        all_rays_rgb = torch.cat(all_rgb, dim=0)
        all_rays_ori = torch.cat(all_ori, dim=0)
        all_rays_dirs = torch.cat(all_dirs, dim=0)

        # Test without optimization
        loss_fn = DistanceBasedScoreLoss()
        (
            _,
            avg_translation_error,
            avg_angular_error,
            avg_score,
            recall,
        ) = test_pose_estimation(
            gs_model,
            scene_info.test_cameras,
            id_module,
            device,
            model_up,
            all_rays_ori,
            all_rays_dirs,
            all_rays_rgb,
            object_id,
            category_name,
            loss_fn,
            False,
            save_rays_dir,
            refine_renderer=refine_renderer,
            refine_matcher=refine_matcher,
            refine_backend_adapter=refine_backend_adapter,
            refine_backend_kwargs=refine_backend_kwargs,
            refine_model=refine_model,
        )

        print("test AVG translation error: ", avg_translation_error)
        print("test AVG angular error: ", avg_angular_error)
        print("test AVG score error: ", avg_score)
        print("test recall: ", recall)

        print("Testing performances with optimization module...")

        # Test with optimization
        (
            test_results,
            test_avg_translation_error,
            test_avg_angular_error,
            _,
            _,
        ) = test_pose_estimation(
            gs_model,
            scene_info.test_cameras,
            id_module,
            device,
            model_up,
            all_rays_ori,
            all_rays_dirs,
            all_rays_rgb,
            object_id,
            category_name,
            None,
            True,
            save_rays_dir,
            refine_renderer=refine_renderer,
            refine_matcher=refine_matcher,
            refine_backend_adapter=refine_backend_adapter,
            refine_backend_kwargs=refine_backend_kwargs,
            refine_model=refine_model,
        )

        print("Optimized AVG translation error: ", test_avg_translation_error)
        print("Optimized AVG angular error: ", test_avg_angular_error)

        return test_results

    except Exception as e:
        print(f"Error during pretraining: {e}")
        traceback.print_exc()
        return None


def evaluate_single_object_in_blender(
        checkpoint_filepath: str,
        checkpoint_args: dotdict,
        exp_dir_filepath: str,
        object_id: str,
        category_name: str,
        starting_seed: int,
        device: str = "cuda",
        lock_backbone: bool = True,
        refine_renderer: str = "2dgs",
        refine_matcher: str = "akaze_gms",
        refine_backend_adapter=None,
        refine_backend_kwargs: "Optional[dict]" = None,
        refine_model_path: str = "",
):
    """Wrapper function for evaluating single object"""
    results = pretrain_single_object(
        checkpoint_filepath,
        checkpoint_args,
        exp_dir_filepath,
        object_id,
        category_name,
        starting_seed=starting_seed,
        device=device,
        lock_backbone=lock_backbone,
        refine_renderer=refine_renderer,
        refine_matcher=refine_matcher,
        refine_backend_adapter=refine_backend_adapter,
        refine_backend_kwargs=refine_backend_kwargs,
        refine_model_path=refine_model_path,
    )
    return results


def main():
    """Main execution function"""
    try:
        args, _ = parse_args()

        # Set reproducibility settings
        use_seed = os.environ.get("USE_SEED", "false").lower() == "true"
        if use_seed:
            seed = int(os.environ.get("SEED", "55176280"))
            print(f"[Reproducibility] Using global seed = {seed}")
            set_global_seed(seed, deterministic=True)
        else:
            print("[Reproducibility] Running without fixed random seed")
            seed = None

        # Create output directory
        out_path_abs = os.path.abspath(args.out_path)
        out_dir = os.path.dirname(out_path_abs)
        os.makedirs(out_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optional: load refinement backend adapter (for custom)
        refine_backend = getattr(args, "refine_renderer", "2dgs")
        refine_matcher = getattr(args, "refine_matcher", "akaze_gms")
        backend_adapter = None
        backend_kwargs = {}
        try:
            backend_kwargs = json.loads(getattr(args, "refine_backend_kwargs", "{}") or "{}")
        except Exception:
            backend_kwargs = {}

        if refine_backend == "custom":
            mod_name = getattr(args, "refine_backend_module", "") or ""
            if mod_name:
                backend_adapter = importlib.import_module(mod_name)
        refine_model_path = getattr(args, "refine_model_path", "") or ""

        # Process all experiments without data type filtering
        results = []
        experiments_to_test = parse_exp_dir(args.exp_path, "")

        print(f"Found {len(experiments_to_test)} experiments to process:")
        for exp_id, exp_info in experiments_to_test.items():
            print(f"  - {exp_id}: {exp_info['exp_dir_filepath']}")

        if len(experiments_to_test) == 0:
            print(f"Warning: No experiments found in {args.exp_path}")
            print("Available directories:")
            if os.path.exists(args.exp_path):
                for item in os.listdir(args.exp_path):
                    item_path = os.path.join(args.exp_path, item)
                    if os.path.isdir(item_path):
                        print(f"  - {item}")
            return

        for experiment_to_test in experiments_to_test.values():
            exp_dir_filepath = experiment_to_test["exp_dir_filepath"]
            checkpoint_filepath = experiment_to_test["checkpoint_filepath"]
            object_id = experiment_to_test["sequence_id"]
            category_name = experiment_to_test["category_name"]
            checkpoint_args = get_checkpoint_arguments(exp_dir_filepath)

            try:
                obj_results = evaluate_single_object_in_blender(
                    checkpoint_filepath,
                    checkpoint_args,
                    exp_dir_filepath,
                    object_id,
                    category_name,
                    starting_seed=seed,
                    lock_backbone=True,
                    device=device,
                    refine_renderer=refine_backend,
                    refine_matcher=refine_matcher,
                    refine_backend_adapter=backend_adapter,
                    refine_backend_kwargs=backend_kwargs,
                    refine_model_path=refine_model_path,
                )
                if obj_results is not None:
                    results.extend(obj_results)
            except RuntimeError as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()

        # Save final results
        print("Saving results")
        with open(out_path_abs, "w") as fh:
            json.dump(results, fh)

    except Exception as e:
        print(f"Error during main execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

