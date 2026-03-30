import os
from cfg_grammar import parse_config


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_checkpoint_arguments(root_dir):
    with open(os.path.join(root_dir, "cfg_args")) as filehandle:
        config_dict = parse_config(filehandle.read())
    return dotdict(config_dict)


def get_highest_valid_checkpoint(root_dir):
    ckpt_dir = os.path.join(root_dir, "point_cloud")

    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return ""

    iteration_dirs = []
    for item in os.listdir(ckpt_dir):
        item_path = os.path.join(ckpt_dir, item)
        if os.path.isdir(item_path) and item.startswith("iteration_"):
            iteration_dirs.append(item)

    if not iteration_dirs:
        print(f"No iteration directories found in {ckpt_dir}")
        return ""

    largest_iteration = -1
    largest_checkpoint_path = ""

    for dir_name in iteration_dirs:
        try:
            iteration_num = int(dir_name.split("_")[1])
            ckpt_filepath = os.path.join(ckpt_dir, dir_name, "point_cloud.ply")

            if os.path.exists(ckpt_filepath) and iteration_num > largest_iteration:
                largest_iteration = iteration_num
                largest_checkpoint_path = ckpt_filepath
        except (ValueError, IndexError):
            continue

    if largest_checkpoint_path:
        print(f"Found checkpoint: {largest_checkpoint_path}")
    else:
        print(f"No valid checkpoint found in {ckpt_dir}")

    return largest_checkpoint_path


def parse_exp_dir(exp_dir, prefix):
    objects_checkpoints = {}

    if os.path.exists(os.path.join(exp_dir, "point_cloud")):
        exp_dir_filepath = exp_dir
        dir_name = os.path.basename(exp_dir)

        if prefix and not dir_name.startswith(prefix):
            return objects_checkpoints

        name_components = dir_name.split("_")
        sequence_id = name_components[-1]
        category_name = "_".join(name_components[:-1]) if len(name_components) > 1 else dir_name

        checkpoint_filepath = get_highest_valid_checkpoint(exp_dir_filepath)
        if checkpoint_filepath:
            objects_checkpoints[sequence_id] = {
                "exp_dir_filepath": exp_dir_filepath,
                "checkpoint_filepath": checkpoint_filepath,
                "sequence_id": sequence_id,
                "category_name": category_name,
            }

    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if not os.path.isdir(item_path):
            continue

        if prefix and not item.startswith(prefix):
            continue

        if os.path.exists(os.path.join(item_path, "point_cloud")):
            exp_dir_filepath = item_path
            name_components = item.split("_")
            sequence_id = name_components[-1]
            category_name = "_".join(name_components[:-1]) if len(name_components) > 1 else item

            checkpoint_filepath = get_highest_valid_checkpoint(exp_dir_filepath)
            if checkpoint_filepath:
                objects_checkpoints[sequence_id] = {
                    "exp_dir_filepath": exp_dir_filepath,
                    "checkpoint_filepath": checkpoint_filepath,
                    "sequence_id": sequence_id,
                    "category_name": category_name,
                }

    return objects_checkpoints