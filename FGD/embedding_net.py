import torch
import torch.nn as nn
import numpy as np


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    """
    Create a convolutional block with optional batch normalization and LeakyReLU activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        downsample (bool): Whether to use downsampling convolution
        padding (int): Padding for convolution
        batchnorm (bool): Whether to use batch normalization

    Returns:
        nn.Sequential: Convolutional block
    """
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class GestureEncoder(nn.Module):
    def __init__(self, gesture_dim, n_frames):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvNormRelu(gesture_dim, 128, batchnorm=True),
            ConvNormRelu(128, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )
        # Dynamically calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.randn(1, n_frames, gesture_dim)
            dummy_features = self.feature_extractor(dummy_input.transpose(1, 2))
            flattened_size = dummy_features.numel()
        # in_channels = 32 * n_frames
        # in_channels = 1248
        self.embedding_network = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32),
        )

    def forward(self, poses):
        poses = poses.transpose(1, 2)  # (bs, seq, dim) to (bs, dim, seq)

        out = self.feature_extractor(poses)
        features_flat = out.view(out.size(0), -1)
        embedding = self.embedding_network(features_flat)

        return embedding


class GestureDecoder(nn.Module):
    def __init__(self, gesture_dim, n_frames):
        super().__init__()

        out_channels = n_frames * 4
        self.reconstruction_network = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Linear(64, out_channels),
        )

        self.gesture_generator = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, gesture_dim, 3),
        )

    def forward(self, feat):
        out = self.reconstruction_network(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.gesture_generator(out)
        out = out.transpose(1, 2)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, gesture_dim, n_frames):
        super().__init__()
        self.gesture_encoder = GestureEncoder(gesture_dim, n_frames)
        self.gesture_decoder = GestureDecoder(gesture_dim, n_frames)

    def forward(self, gestures):
        gesture_embedding = self.gesture_encoder(gestures)
        output_gesture = self.gesture_decoder(gesture_embedding)
        return gesture_embedding, output_gesture


if __name__ == '__main__':  # model test
    n_frames = 88
    gesture_dim = 1141
    batch_size = 1067
    gesture = torch.randn((batch_size, n_frames, gesture_dim))

    # encoder = GestureEncoder(gesture_dim, n_frames)
    # decoder = GestureDecoder(gesture_dim, n_frames)
    #
    # feat = encoder(gesture)
    # recon_poses = decoder(feat)
    #
    # print('input', gesture.shape)
    # print('feat', feat.shape)
    # print('output', recon_poses.shape)

    model = EmbeddingNet(gesture_dim, n_frames)
    gesture_embedding, output_gesture = model(gesture)
    print(gesture_embedding.shape)
    print(output_gesture.shape)
