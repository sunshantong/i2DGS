import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.graphics_utils import fov2focal
from pose_estimation.distance_based_loss import DistanceBasedScoreLoss
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.test import test_pose_estimation
from pose_estimation.ray_precaching import preprocess_rays_for_training, white_bg_mask_hwc


def train_id_module(
        ckpt_path,
        device,
        id_module: IdentificationModule,
        scene_info,
        sequence_id,
        category_id,
        model,
        pipe,
        bg_color,
        start_iterations: int = 0,
        display_every_n_iterations: int = 20,
        val_every_n_iterations: int = 100,
        n_iterations: int = 1500,
        gradient_accumulation_steps: int = 32,
        lock_backbone: bool = True,
        save_rays_dir: str = ""):
    all_rays_ori, all_rays_dirs, all_rays_rgb = preprocess_rays_for_training(
        scene_info, model, pipe, bg_color, device, save_rays_dir
    )

    # Precompute valid camera indices
    valid_camera_indices = []
    for i in range(len(scene_info.train_cameras)):
        frame_dir = os.path.join(save_rays_dir, f"{i:05d}")
        if os.path.exists(os.path.join(frame_dir, "ori.npy")):
            valid_camera_indices.append(i)

    if len(valid_camera_indices) == 0:
        raise ValueError("No valid camera data found with ray cache!")

    id_module.train()
    if lock_backbone:
        id_module.backbone_wrapper.eval()

    optimizer = AdamW([
        {"params": id_module.backbone_wrapper.parameters(), "lr": 5e-5},
        {"params": id_module.ray_preprocessor.parameters(), "lr": 1e-3},
        {"params": id_module.attention.parameters(), "lr": 1e-3},
        {"params": id_module.camera_direction_prediction_network.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-5, 1e-3, 1e-3, 1e-3],
        total_steps=n_iterations,
        pct_start=0.2,
        anneal_strategy='cos',
        cycle_momentum=False,
    )

    loss_fn = DistanceBasedScoreLoss()
    writer = SummaryWriter()
    writer.add_text("config/sequence_id", sequence_id)
    writer.add_text("config/category_id", category_id)
    writer.add_text("config/ckpt_path", ckpt_path)

    model_up_np = np.mean(
        np.asarray([cam.R[:3, 1] for cam in scene_info.train_cameras], dtype=np.float32),
        axis=0
    )
    model_up_np = model_up_np / (np.linalg.norm(model_up_np) + 1e-8)
    model_up = torch.from_numpy(model_up_np).to(device=device, non_blocking=True)

    running_loss = 0.0

    for iteration in range(start_iterations, n_iterations):
        optimizer.zero_grad()

        # Sample from valid cameras only
        img_idx = torch.randint(
            0, len(valid_camera_indices),
            (gradient_accumulation_steps,),
            dtype=torch.long, device=device
        )

        accumulation_loss = 0.0
        accumulation_cam_up = 0.0
        accumulation_scores_loss = 0.0
        valid_steps = 0

        for gradient_accumulation_step in range(gradient_accumulation_steps):
            valid_idx = img_idx[gradient_accumulation_step].item()
            idx = valid_camera_indices[valid_idx]  # Map to original camera index

            gradient_camera_info = scene_info.train_cameras[idx]

            tensor_image = torch.from_numpy(np.array(gradient_camera_info.image))
            train_img = tensor_image.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0

            if train_img.shape[-1] == 4:
                alpha = train_img[..., -1] > 0.3
                train_img_mask = alpha
                train_img = train_img[..., :3] * alpha.unsqueeze(-1) + (1 - alpha).unsqueeze(-1)
            else:
                train_img_mask = white_bg_mask_hwc(train_img[..., :3]).to(device)

            w2c = torch.eye(4, dtype=torch.float32, device=device)
            w2c[:3, :3] = torch.transpose(torch.from_numpy(gradient_camera_info.R), -1, -2).to(device)
            w2c[:3, -1] = torch.from_numpy(gradient_camera_info.T).to(device)
            c2w = torch.inverse(w2c)
            target_camera_pose = c2w

            focalX = fov2focal(gradient_camera_info.FovX, gradient_camera_info.width)
            focalY = fov2focal(gradient_camera_info.FovY, gradient_camera_info.height)
            target_camera_intrinsic = torch.tensor(
                [[focalX, 0.0, gradient_camera_info.width / 2.0],
                 [0.0, focalY, gradient_camera_info.height / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=device
            )

            frame_dir_idx = os.path.join(save_rays_dir, f"{idx:05d}")

            try:
                rays_ori = torch.from_numpy(np.load(os.path.join(frame_dir_idx, "ori.npy"))).to(device=device,
                                                                                                dtype=torch.float32)
                rays_dirs = torch.from_numpy(np.load(os.path.join(frame_dir_idx, "dir.npy"))).to(device=device,
                                                                                                 dtype=torch.float32)
                rays_rgb = torch.from_numpy(np.load(os.path.join(frame_dir_idx, "color.npy"))).to(device=device,
                                                                                                  dtype=torch.float32)

                finite = torch.isfinite(rays_ori).all(dim=1) & torch.isfinite(rays_dirs).all(dim=1) & torch.isfinite(
                    rays_rgb).all(dim=1)
                rays_ori, rays_dirs, rays_rgb = rays_ori[finite], rays_dirs[finite], rays_rgb[finite]

                if rays_ori.shape[0] == 0:
                    continue

                scores, attn_map, img_features, camera_up_dir, rays_idx = id_module(
                    train_img, train_img_mask, rays_ori, rays_dirs, rays_rgb
                )

                loss_score, target_scores = loss_fn(
                    scores,
                    target_camera_pose,
                    target_camera_intrinsic,
                    rays_ori[rays_idx],
                    rays_dirs[rays_idx],
                    attn_map.shape[-2],
                    id_module.backbone_wrapper.backbone_wh,
                    model_up=model_up,
                )

                cam_up_similarity = -0.5 * torch.cosine_similarity(model_up, camera_up_dir, dim=-1) + 0.5
                combined_loss_score = loss_score + 1.5 * cam_up_similarity

                if not torch.isfinite(combined_loss_score):
                    if iteration < 5:
                        print(f"[diag] non-finite loss at iter {iteration}")
                    continue

                loss = combined_loss_score / gradient_accumulation_steps
                loss.backward()

                accumulation_loss += combined_loss_score.item()
                accumulation_cam_up += cam_up_similarity.item() / gradient_accumulation_steps
                accumulation_scores_loss += loss_score.item() / gradient_accumulation_steps
                valid_steps += 1

            except Exception as e:
                print(f"Error processing camera {idx}: {e}")
                continue

        if valid_steps > 0:
            # Adjust gradient accumulation for actual valid steps
            if valid_steps < gradient_accumulation_steps:
                for param in id_module.parameters():
                    if param.grad is not None:
                        param.grad.data *= (gradient_accumulation_steps / valid_steps)

            torch.nn.utils.clip_grad_norm_(id_module.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            writer.add_scalar("train/loss", accumulation_loss, global_step=iteration)
            writer.add_scalar("train/cam_up", accumulation_cam_up, global_step=iteration)
            writer.add_scalar("train/loss_score", accumulation_scores_loss, global_step=iteration)

            running_loss += accumulation_loss

        if iteration % display_every_n_iterations == display_every_n_iterations - 1:
            last_loss = running_loss / display_every_n_iterations
            print(f"[{iteration}] loss: {last_loss:.6f}")
            running_loss = 0.0

        if iteration % val_every_n_iterations == val_every_n_iterations - 1:
            print("Evaluating on training set...")
            try:
                (_, train_avg_translation_error,
                 train_avg_angular_error,
                 train_avg_score,
                 train_recall) = test_pose_estimation(
                    model, scene_info.train_cameras, id_module, device, model_up,
                    all_rays_ori, all_rays_dirs, all_rays_rgb,
                    sequence_id, category_id, loss_fn,
                    False, save_rays_dir
                )

                writer.add_scalar("train/avg_translation_error", train_avg_translation_error, global_step=iteration)
                writer.add_scalar("train/avg_angular_error", train_avg_angular_error, global_step=iteration)
                writer.add_scalar("train/avg_loss_score", train_avg_score, global_step=iteration)
                writer.add_scalar("train/recall", train_recall, global_step=iteration)
            except Exception as e:
                print(f"Error in training set evaluation: {e}")

            print("Evaluating on validation set...")
            try:
                (_, val_avg_translation_error,
                 val_avg_angular_error,
                 val_avg_score,
                 val_recall) = test_pose_estimation(
                    model, scene_info.test_cameras, id_module, device, model_up,
                    all_rays_ori, all_rays_dirs, all_rays_rgb,
                    sequence_id, category_id, loss_fn,
                    False, save_rays_dir
                )

                writer.add_scalar("val/avg_translation_error", val_avg_translation_error, global_step=iteration)
                writer.add_scalar("val/avg_angular_error", val_avg_angular_error, global_step=iteration)
                writer.add_scalar("val/avg_loss_score", val_avg_score, global_step=iteration)
                writer.add_scalar("val/recall", val_recall, global_step=iteration)

            except Exception as e:
                print(f"Error in validation set evaluation: {e}")

            id_module.train()
            if lock_backbone:
                id_module.backbone_wrapper.eval()

    # Save final model
    torch.save({
        "epoch": n_iterations,
        "model_state_dict": id_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "running_loss": running_loss,
    }, ckpt_path)
    print(f"Training finished. Final model saved to {ckpt_path}")
    writer.close()
