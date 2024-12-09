import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from concurrent.futures import ProcessPoolExecutor

from argparse import ArgumentParser

import glob
import os
import sys
import joblib as jl
import glob

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import pymo
from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

target_joints = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head", "HeadEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandPinky1", "RightHandPinky2",
                 "RightHandPinky3", "RightHandPinky4", "RightForeArmEnd", "RightArmEnd", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
                 "LeftHandPinky4", "LeftForeArmEnd", "LeftArmEnd", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBaseEnd", "RightLegEnd", "RightUpLegEnd", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBaseEnd", "LeftLegEnd", "LeftUpLegEnd"]


def process_file(file):
    print("Processing", file)
    p = BVHParser()
    return p.parse(file)


def extract_joint_angles(bvh_dir, files, npy_dir, pipeline_dir, fps):
    p = BVHParser()

    data_all = list()
    for f in files:
        print("Processing ", f)
        bvh_parsed = p.parse(f)
        # frames = bvh_parsed.skeleton[:10]
        # frames = bvh_parsed.values[:10]
        data_all.append(bvh_parsed)

    # with ProcessPoolExecutor() as executor:
    #     data_all = list(executor.map(process_file, files))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        # ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        ('position', MocapParameterizer('position')),
        # ('slicer', Slicer(88)),
        # ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)

    print(np.shape(out_data))

    # the data pipe will append the mirrored files to the end
    # assert len(out_data) == len(files)

    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))

    for i, f in enumerate(files):
        ff = f.split("/")[-1]
        print("Saving npy: ", ff)
        npy_file = os.path.join(npy_dir, ff[:-4] + ".npy")
        np.save(npy_file, out_data[i])


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
    parser.add_argument('--pipeline_dir', '-pipe', default="./processed/zeggs",
                        help="Path where the motion data processing pipeline will be stored")

    args = parser.parse_args()

    print("Going to pre-process the following motion files:")
    files = sorted([f for f in glob.iglob(args.bvh_dir + '/*.bvh')])[:2]
    print("Total files: {}".format(len(files)))

    extract_joint_angles(args.bvh_dir, files, args.npy_dir, args.pipeline_dir, fps=60)
