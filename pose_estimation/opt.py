import configargparse


def parse_args():
    parser = configargparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        default="./log",
        help="experiment directory",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        default="pose_eval.json",
        help="experiment directory",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["blender", "mip360", "tankstemple", "cambridge_landmark", "all"],
        default="all",
        help="the type of data to validate",
    )

    # Pose refinement (PnP) renderer backend selection
    parser.add_argument(
        "--refine_renderer",
        type=str,
        choices=["2dgs", "custom"],
        default="2dgs",
        help="renderer backend used only in pose refinement stage",
    )
    parser.add_argument(
        "--refine_matcher",
        type=str,
        choices=["akaze_gms", "sift", "lightglue", "loftr"],
        default="akaze_gms",
        help="feature matcher used in pose refinement stage",
    )
    parser.add_argument(
        "--refine_backend_module",
        type=str,
        default="",
        help="python module path for custom renderer adapter (e.g. prerender_adapter)",
    )
    parser.add_argument(
        "--refine_backend_kwargs",
        type=str,
        default="{}",
        help="json dict string passed to backend adapter (e.g. '{\"ckpt\":\"...\"}')",
    )
    parser.add_argument(
        "--refine_model_path",
        type=str,
        default="",
        help="path to model used by refine renderer. If empty, defaults to 2DGS ply.",
    )

    args, extras = parser.parse_known_args()
    return args, extras

