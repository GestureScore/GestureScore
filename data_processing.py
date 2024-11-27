from argparse import ArgumentParser
import os
import numpy as np
import torch


def load_bvh(bvh_dir):
    from anim import bvh

    bvh_files = [bvh_file for bvh_file in sorted(os.listdir(bvh_dir)) if bvh_file.endswith('.bvh')]

    assert len(bvh_files) > 0, 'No .bvh files found in {}'.format(bvh_dir)

    gesture_list = list()
    names = list()
    for file in bvh_files:
        names.append(file)

        anim_data = bvh.load(os.path.join(bvh_dir, file))
        positions = np.asarray(anim_data['positions'], dtype=np.float32)
        np_poistion = positions.reshape(positions.shape[0], -1)
        gesture_list.append(np_poistion)

    gesture_bvh = np.array(gesture_list, np.float32)
    return gesture_bvh, names


def sample_all_frame_to_clips(real_data_list, predict_data_list, names, n_frame):
    real_clip = []
    predict_clip = []

    for name, real_data, predict_data in zip(names, real_data_list, predict_data_list):
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
    python data_processing.py --real_path="./data/zeggs/real" --predict_path="./data/zeggs/predict"
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
    real_bvh, names = load_bvh(args.real_path)
    real_data = np.asarray(real_bvh).to(device)
    print("Real gesture shape ", np.shape(real_data))

    print("Loading predict bvh files {!r}".format(args.predict_path))
    predict_bvh, _ = load_bvh(args.predict_path)
    predict_data = torch.Tensor(predict_bvh).to(device)
    print("Predict gesture shape ", np.shape(predict_data))

    assert len(real_data) == len(predict_data), "Real data and predict data do not match"

    real_clip, predict_clip = sample_all_frame_to_clips(real_data, predict_data, names, n_frame)

    print(f"real_clip: {np.shape(real_clip)}")
    print(f"predict_clip: {np.shape(predict_clip)}")

    np.savez_compressed('./data/real_dataset.npz', gesture=real_clip)
    np.savez_compressed('./data/predict_dataset.npz', gesture=predict_clip)
