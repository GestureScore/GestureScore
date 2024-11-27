import numpy as np
import torch
from scipy import linalg


class EmbeddingSpaceEvaluator:
    def __init__(self, model_embedding, gesture_dim, n_frames):
        # data feature
        self.pose_dim = gesture_dim
        self.model_embedding = model_embedding

        # sample data
        self.real_samples = []
        self.generate_samples = []
        self.real_feat_list = []
        self.generated_feat_list = []

    def reset(self):
        self.real_samples = []
        self.generate_samples = []
        self.real_feat_list = []
        self.generated_feat_list = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_real_samples(self, samples):
        feat, _ = self.model_embedding(samples)

        self.real_samples.append(samples.cpu().numpy().reshape(samples.shape[0], -1))
        self.real_feat_list.append(feat.data.cpu().numpy())

    def push_generated_samples(self, samples):
        feat, _ = self.model_embedding(samples)

        self.generate_samples.append(samples.cpu().numpy().reshape(samples.shape[0], -1))
        self.generated_feat_list.append(feat.data.cpu().numpy())

    def get_fgd(self, use_feat_space=True):
        if use_feat_space:  # on feature space
            generated_data = np.vstack(self.generated_feat_list)
            real_data = np.vstack(self.real_feat_list)
        else:  # on raw pose space
            generated_data = np.vstack(self.generate_samples)
            real_data = np.vstack(self.real_samples)

        frechet_dist = self.frechet_distance(generated_data, real_data)
        return frechet_dist

    def frechet_distance(self, samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':
    from embedding_net import EmbeddingNet

    device = torch.device("mps")

    batch_size = 1067
    n_frame = 88
    gesture_dim = 1141

    # model
    model_embedding = EmbeddingNet(gesture_dim, n_frame)
    model_embedding.to(device)

    # load model
    model_embedding_path = f"./output/model_checkpoint_{gesture_dim}_{n_frame}.bin"
    checkpoint = torch.load(model_embedding_path, map_location=device, weights_only=True)

    model_embedding.load_state_dict(checkpoint['embedding_model'])
    model_embedding.eval()

    model = EmbeddingSpaceEvaluator(model_embedding, gesture_dim, n_frame)
    gesture_1 = np.random.rand(batch_size, n_frame, gesture_dim).astype(np.float32)
    gesture_2 = np.random.rand(batch_size, n_frame, gesture_dim).astype(np.float32)

    g1 = np.vstack(gesture_1)
    g2 = np.vstack(gesture_2)
    print(g1.shape, g2.shape)

    mean_1 = np.mean(g1)
    variance_1 = np.var(g1)

    mean_2 = np.mean(g2)
    variance_2 = np.var(g2)

    fid = model.calculate_frechet_distance(mean_1, variance_1, mean_2, variance_2)
    print(fid)

    fid_dis = model.frechet_distance(g1, g2)
    print(fid_dis)
