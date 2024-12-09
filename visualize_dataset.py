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


def visualize_sample(name, data, join_name, plot_dir="./plot"):
    file_path = os.path.join(plot_dir, name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Plotting
    plt.figure(figsize=(12, 6))
    for axis in range(len(axis_name)):
        plt.plot(data[:, axis], label=f'{axis_name[axis]}')

    plt.legend(loc='upper right')
    plt.title(f'Join {join_name}')
    plt.savefig(os.path.join(file_path, f'{name}_{join_name}.png'))
    print("Saved figure {}".format(os.path.join(file_path, f'{name}_{join_name}.png')))

    # plt.show()


if __name__ == '__main__':
    """
    python bvh2npy.py --bvh_dir="./data/zeggs/predict" --npy_dir="./processed/zeggs/predict" --pipeline_dir="./processed/zeggs"
    """
    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--bvh_dir', '-bvh', required=True, default="./data/zeggs/predict",
                        help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--npy_dir', '-npy', required=True, default="./processed/zeggs/predict",
                        help="Path where extracted motion features will be stored")

    args = parser.parse_args()

    print("Pre-process the following motion files:")
    files = [f for f in sorted(glob.iglob(args.bvh_dir + '/*.bvh'))][:2]
    print("Total files: {}".format(len(files)))

    for bvh_file in files:
        name = bvh_file.split("/")[-1].split(".")[0]
        print("Processing {}".format(name))

        anim_data = bvh.load(bvh_file)
        rotations = anim_data['rotations']
        rotation_flatted = np.reshape(rotations, (rotations.shape[0], 225))
        print(rotations.shape)
        print(rotation_flatted.shape)

        subdivision = rotation_flatted[:, :]

        # for r in range(rotations.shape[1]):
        #     visualize_sample(name=name, data=rotations[:100, r, :], join_name=joint_names[r], plot_dir="./plot/real")
