import numpy as np
from argparse import ArgumentParser
import joblib as jl
import glob
import pdb
import os
from anim import bvh, quat, txform
import torch
import matplotlib.pyplot as plt

joint_names = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head", "HeadEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandPinky1", "RightHandPinky2",
                 "RightHandPinky3", "RightHandPinky4", "RightForeArmEnd", "RightArmEnd", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
                 "LeftHandPinky4", "LeftForeArmEnd", "LeftArmEnd", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBaseEnd", "RightLegEnd", "RightUpLegEnd", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBaseEnd", "LeftLegEnd", "LeftUpLegEnd"]
axis_name = ["X", "Y", "Z"]


# def visualize_sample(name, real_data, predict_data, join_name, plot_dir="./plot"):
#     file_path = os.path.join(plot_dir, name)
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#
#     # Plotting
#     plt.figure(figsize=(12, 6))
#     for axis in range(len(axis_name)):
#         plt.plot(data[:, axis], label=f'{axis_name[axis]}')
#
#     plt.legend(loc='upper right')
#     plt.title(f'Join {join_name}')
#     plt.savefig(os.path.join(file_path, f'{name}_{join_name}.png'))
#     print("Saved figure {}".format(os.path.join(file_path, f'{name}_{join_name}.png')))
#
#     # plt.show()
def visualize_sample(name, real_data, predict_data, join_name, plot_dir="./plot"):
    file_path = os.path.join(plot_dir, name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Plotting
    plt.figure(figsize=(14, 8))

    # Real data subplot
    plt.subplot(1, 2, 1)
    for axis in range(real_data.shape[1]):
        plt.plot(real_data[:, axis], label=f'Real {axis_name[axis]}')
    plt.legend(loc='upper right')
    plt.title(f'Real Data - {join_name}')

    # Predicted data subplot
    plt.subplot(1, 2, 2)
    for axis in range(predict_data.shape[1]):
        plt.plot(predict_data[:, axis], label=f'Predicted {axis_name[axis]}')
    plt.legend(loc='upper right')
    plt.title(f'Predicted Data - {join_name}')

    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{name}_{join_name}.png'))
    print(f"Saved figure {os.path.join(file_path, f'{name}_{join_name}.png')}")


if __name__ == '__main__':
    """
    python visualize_dataset.py --predict="./data/predict_dataset.npz" --real="./data/real_dataset.npz"
    """
    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--predict', '-predict]', required=True, default="./data/predict_dataset.npz",
                        help="Path to predict dataset")
    parser.add_argument('--real', '-real', required=True, default="./data/real_dataset.npz",
                        help="Path to real dataset")

    args = parser.parse_args()

    item_idx = 1

    # real = np.load(args.real)
    predict_dataset = np.load(args.predict)["gesture"]
    predict_sample = predict_dataset[item_idx]
    rotations_predict = np.reshape(predict_sample, (predict_sample.shape[0], 75, 3))

    real_dataset = np.load(args.real)["gesture"]
    real_sample =  real_dataset[item_idx]
    rotations_real = np.reshape(real_sample, (real_sample.shape[0], 75, 3))

    assert rotations_real.shape == rotations_predict.shape, "Shapes do not match"
    print("Shape: ", rotations_predict.shape)


    for r in range(rotations_real.shape[1]):
        visualize_sample(name=str(item_idx), real_data=rotations_real[:, r, :], predict_data=rotations_predict[:, r, :], join_name=joint_names[r], plot_dir="./plot/compare")

    # for bvh_file in files:
    #     name = bvh_file.split("/")[-1].split(".")[0]
    #     print("Processing {}".format(name))
    #
    #     anim_data = bvh.load(bvh_file)
    #     rotations = anim_data['rotations']
    #     rotation_flatted = np.reshape(rotations, (rotations.shape[0], 225))
    #     print(rotations.shape)
    #     print(rotation_flatted.shape)
    #
    #     subdivision = rotation_flatted[:, :]
    #
    #     # for r in range(rotations.shape[1]):
    #     #     visualize_sample(name=name, data=rotations[:100, r, :], join_name=joint_names[r], plot_dir="./plot/real")
