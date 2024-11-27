import os

import numpy as np
import torch
import argparse
from embedding_space_evaluator import EmbeddingSpaceEvaluator


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(gt_data)
    fgd_evaluator.push_generated_samples(test_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat, fdg_on_raw


def main(args, n_frame, gesture_dim):
    # AE model
    ae_path = f'output/model_checkpoint_{gesture_dim}_{n_frame}.bin'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)

    # load GT data
    gt_data = make_tensor(f'../data/{tier}/{code}NA', chunk_len).to(device)

    # load generated data
    generated_data_path = f'../data/{tier}'
    folders = sorted([f.path for f in os.scandir(generated_data_path) if f.is_dir()])

    print(f'----- Experiment (motion chunk length: {chunk_len}) -----')
    print('FGDs on feature space and raw data space')
    for folder in folders:
        test_data = make_tensor(folder, chunk_len).to(device)
        fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
        print(f'{os.path.basename(folder)}: {fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
    print("Finish")


if __name__ == '__main__':
    """
    python evaluate_FGD
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dataset', '-real', required=True, default="../data/real_dataset.npz",
                        help="")
    parser.add_argument('--predict_dataset', '-predict', required=True, default="../data/predict_dataset.npz",
                        help="")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.gpu)

    n_frame = 88
    gesture_dim = 1141

    main(args, n_frame, gesture_dim)
