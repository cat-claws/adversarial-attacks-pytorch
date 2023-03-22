import torch
import torch.nn as nn

from ..attack import Attack
from scipy.stats import binomtest
from transformers.utils import ModelOutput

class BruteForceUniform(Attack):
    r"""
    BruteForceUniform in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BruteForceUniform(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255, alpha = 5/100, mu = 0.95, pop=128):
        super().__init__("BruteForceUniform", model)
        self.eps = eps
        self.alpha = alpha
        self.pop = pop
        self.mu = mu
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        rejected = torch.zeros_like(labels).bool()

        K = torch.zeros_like(labels)
        N = torch.zeros_like(labels)

        with torch.no_grad():

            while not rejected.all():
                x = images[~rejected]
                xs = x.repeat(self.pop // len(x), 1, 1, 1, 1)
                ub = torch.clamp(x + self.eps, min = 0, max = 1)
                lb = torch.clamp(x - self.eps, min = 0, max = 1)
                xs = (ub - lb) * torch.rand_like(xs) + lb

                outputs = self.get_logits(xs.view(-1, *xs.shape[2:])).view(xs.size(0), xs.size(1), -1)
                K.masked_scatter_(~rejected, K[~rejected] + (outputs.argmax(-1) == labels[~rejected]).sum(dim = 0))
                N.masked_scatter_(~rejected, N[~rejected] + len(outputs))

                for i in range(len(images)):
                    if not rejected[i]:
                        rejected[i] = bool(binomtest(K[i], N[i], p=self.mu, alternative='two-sided').pvalue < self.alpha)


        return ModelOutput(rejected = rejected, K = K, N = N)
