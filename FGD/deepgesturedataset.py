import os

import torch
from torch.utils.data import Dataset
import numpy as np


class DeepGestureDataset(Dataset):
    def __init__(self, dataset_file):
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(dataset_file)

        self.dataset_file = dataset_file
        gesture_clips = np.load(self.dataset_file)
        self.gestures = gesture_clips['gesture']

        self.n_frame, self.feature_dim = np.shape(self.gestures[0])
        self.total = len(self.gestures)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        return self.gestures[index]


if __name__ == '__main__':
    # dataset = DeepGestureDataset(dataset_file='../data/real_dataset.npz')
    # print(len(dataset))
    # print(dataset[0].shape)

    dataset = DeepGestureDataset(dataset_file='../data/predict_dataset.npz')
    print(len(dataset))
    print(dataset[0].shape)
