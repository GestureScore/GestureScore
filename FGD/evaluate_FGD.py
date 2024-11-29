import os
import numpy as np
import torch
import argparse


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(gt_data)
    fgd_evaluator.push_generated_samples(test_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat, fdg_on_raw


def load_embedding_model(args, gesture_dim, n_frame, device):
    from embedding_net import EmbeddingNet

    print("Loading embedding model at path: {}.".format(args.path))
    # model
    model_embedding = EmbeddingNet(gesture_dim, n_frame)
    model_embedding.to(device)

    # load model
    checkpoint = torch.load(args.model_embedding_path, map_location=device, weights_only=True)
    model_embedding.load_state_dict(checkpoint['embedding_model'])
    model_embedding.eval()

    return model_embedding


def main(args, gesture_dim, n_frame, device):
    from embedding_space_evaluator import EmbeddingSpaceEvaluator
    from deepgesturedataset import DeepGestureDataset

    model_embedding = load_embedding_model(args, gesture_dim, n_frame, device)
    fgd_evaluator = EmbeddingSpaceEvaluator(model_embedding, gesture_dim, n_frame)

    print("Loading ground truth dataset at path {}.".format(args.path))
    real_dataset = DeepGestureDataset(dataset_file=args.real_dataset)
    real_data = torch.Tensor(real_dataset.get_all()).to(device)

    print("Loading predicted dataset at path {}.".format(args.path))
    predict_dataset = DeepGestureDataset(dataset_file=args.predict_dataset)
    predict_data = torch.Tensor(predict_dataset.get_all()).to(device)

    print("Evaluating FGD...")
    fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, real_data, predict_data)
    print(f'{fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')

    # print(f'----- Experiment (motion chunk length: {chunk_len}) -----')
    # print('FGDs on feature space and raw data space')
    # for folder in folders:
    #     test_data = make_tensor(folder, chunk_len).to(device)
    #     fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
    #     print(f'{os.path.basename(folder)}: {fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
    # print("Finish")


if __name__ == '__main__':
    """
    python evaluate_FGD -real=../data/real_dataset.npz -predict ../data/predict_dataset.npz --gpu=mps
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dataset', '-real', required=True, default="../data/real_dataset.npz",
                        help="")
    parser.add_argument('--predict_dataset', '-predict', required=True, default="../data/predict_dataset.npz",
                        help="")
    parser.add_argument('--model_embedding_path', '-model', default="./output/model_checkpoint_1141_88.bin",
                        help="")
    parser.add_argument('--gpu', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.gpu)

    n_frame = 88
    # gesture_dim = 1141
    gesture_dim = 225
    args.model_embedding_path = f"./output/embedding_network_{gesture_dim}_{n_frame}.pth"

    main(args, gesture_dim, n_frame, device)
