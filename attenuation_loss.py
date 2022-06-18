import torch
import torch.nn.functional as F
from torch.nn.modules import Module


class AttenuationLoss(Module):
    r"""PyTorch implementation of the learned loss attentuation for Bayesian classification as presented by
        Kendall and Gal (2017).

        Computes a numerically stable approximation of the expected log likelihood of a Gaussian using Monte Carlo
        sampling.
        While the arguments require the log standard deviation, any tensor can be passed which is interpreted as the
        log standard deviation. The loss function will be optimized to learn the log standard deviation.

        Args:
            eps (float, optional): Value used to numerical for stability. Default: 1e-8.
            num_samples (int, optional): Number of samples to use for approximating the expected log likelihood.
            Default: 10.
            device (torch.device, optional): Device on which the loss is computed. Default: torch.device('cpu').

        Forward pass args:
            - mu (tensor): Gaussian distribution mean over pixels of image batches.
            - log_sigma: Gaussian distribution log standard deviation over pixels of image batches.
            - target: Target image batches.
            - output: Scalar loss value.
        """
    def __init__(self, *, eps: float = 1e-8, num_samples: int = 10, device=torch.device('cpu')) -> None:
        super(AttenuationLoss, self).__init__()
        self.eps = eps
        self.T = num_samples
        self.device = device

    def forward(self, mu: torch.tensor, log_sigma: torch.tensor, target: torch.tensor) -> torch.tensor:
        return attenuation_loss(mu, log_sigma, target, eps=self.eps, T=self.T, device=self.device)


def attenuation_loss(mu: torch.tensor, log_sigma: torch.tensor, target: torch.tensor, eps: float = 1e-8, T: int = 10,
             device=torch.device('cpu')):
    # Compute standard deviation from the log standard deviation to sample from the Gaussian.
    sigma = torch.exp(log_sigma) + eps

    logit_sum = torch.zeros((target.size(0), 10, target.size(1), target.size(2)), device=device)
    # Sample T logits by corrupting the predicted mu with Gaussian noise with standard deviation sigma.
    for _ in range(T):
        logit_sum += torch.softmax(mu + torch.mul(sigma, torch.randn(sigma.size(), device=device)), dim=1)

    # Computing the negative log likelihood of the mean logit. Adding eps to avoid log(0).
    return F.nll_loss(torch.log(logit_sum / T + eps), target, reduction="mean")
