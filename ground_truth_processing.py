from argparse import ArgumentParser
import os
import numpy as np
import torch
from anim import bvh


def load_bvh(bvh_dir):
    bvh_files = [bvh_file for bvh_file in sorted(os.listdir(bvh_dir)) if bvh_file.endswith('.bvh')]

    assert len(bvh_files) > 0, 'No .bvh files found in {}'.format(bvh_dir)

    gesture_list = dict()
    for file in bvh_files:
        print("Loading {}".format(os.path.join(bvh_dir, file)))
        anim_data = bvh.load(os.path.join(bvh_dir, file))
        positions = np.asarray(anim_data['rotations'], dtype=np.float32)
        position_flattened = positions.reshape(positions.shape[0], -1)
        gesture_list[file] = position_flattened

    return gesture_list


def sample_all_frame_to_clips(real_data_dict, n_frame):
    gt_clip = []

    for name in real_data_dict.keys():
        gt_data = real_data_dict[name]

        print(f"{name}: [{np.shape(gt_data)}]")

        length = gt_data.shape[0]
        n_subdivision = int(length // n_frame)

        print(f"{length} -> {n_subdivision} * {n_frame}")

        for division in range(n_subdivision):
            gt_clip.append(gt_data[division * n_frame:(division + 1) * n_frame])

    return gt_clip


if __name__ == '__main__':
    """
    python data_processing.py --ground_truth_path=./data/zeggs/ground_truth
    """
    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--ground_truth_path', '-gt', required=True, default="./data/zeggs/ground_truth", help="")
    args = parser.parse_args()
    n_frame = 88

    print("Loading real bvh files {!r}".format(args.ground_truth_path))
    real_bvh_dict = load_bvh(args.ground_truth_path)
    print("Total real gesture ", len(real_bvh_dict))

    gt_clip = sample_all_frame_to_clips(real_bvh_dict, n_frame)

    print(f"real_clip: {np.shape(gt_clip)}")

    np.savez_compressed('./data/ground_truth_dataset.npz', gesture=gt_clip)
    print("Saved ./data/ground_truth_dataset.npz")
