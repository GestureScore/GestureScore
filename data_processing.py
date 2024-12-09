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


def sample_all_frame_to_clips(real_data_dict, predict_data_dict, n_frame):
    real_clip = []
    predict_clip = []

    for name in real_data_dict.keys():
        real_data = real_data_dict[name]
        predict_data = predict_data_dict[name]

        print(f"{name}: [{np.shape(real_data)}], [{np.shape(predict_data)}]")

        length = min(real_data.shape[0], predict_data.shape[0])
        n_subdivision = int(length // n_frame)

        print(f"{length} -> {n_subdivision} * {n_frame}")

        for division in range(n_subdivision):
            real_clip.append(real_data[division * n_frame:(division + 1) * n_frame])
            predict_clip.append(predict_data[division * n_frame:(division + 1) * n_frame])

    return real_clip, predict_clip


if __name__ == '__main__':
    """
    python data_processing.py --real_path=./data/zeggs/real --predict_path=./data/zeggs/predict --gpu=cuda:0
    """
    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--real_path', '-real', required=True, default="./data/zeggs/real",
                        help="")
    parser.add_argument('--predict_path', '-predict', required=True, default="./data/zeggs/predict",
                        help="")
    parser.add_argument('--gpu', '-gpu', required=True, default="cuda:0",
                        help="")
    args = parser.parse_args()
    device = torch.device(args.gpu)
    n_frame = 88

    print("Loading real bvh files {!r}".format(args.real_path))
    real_bvh_dict = load_bvh(args.real_path)
    print("Total real gesture ", len(real_bvh_dict))

    print("Loading predict bvh files {!r}".format(args.predict_path))
    predict_bvh_dict = load_bvh(args.predict_path)
    print("Total predict gesture ", len(predict_bvh_dict))

    assert len(real_bvh_dict) == len(predict_bvh_dict), "Real data and predict data do not match"

    real_clip, predict_clip = sample_all_frame_to_clips(real_bvh_dict, predict_bvh_dict, n_frame)

    print(f"real_clip: {np.shape(real_clip)}")
    print(f"predict_clip: {np.shape(predict_clip)}")

    np.savez_compressed('./data/real_dataset.npz', gesture=real_clip)
    print("Saved ./data/real_dataset.npz")
    np.savez_compressed('./data/predict_dataset.npz', gesture=predict_clip)
    print("Saved ./data/predict_dataset.npz")
