import torch
import torch.nn as nn

from ..attack import Attack


class Alion(Attack):

    def __init__(self, model, eps=8/255,
                 rho=0.9, steps=10, random_start=True):
        super().__init__("Alion", model)
        self.eps = eps
        # self.alpha = alpha
        self.rho = rho
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # add here
        square_avg = torch.zeros_like(adv_images)
        acc_delta = torch.zeros_like(adv_images)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            square_avg.mul_(self.rho).addcmul_(grad, grad, value=1 - self.rho)
            std = square_avg.add(1e-7).sqrt_()
            delta = acc_delta.add(1e-7).sqrt_().div_(std).mul_(grad)
            acc_delta.mul_(self.rho).addcmul_(delta, delta, value=1 - self.rho)

            adv_images = adv_images.detach() + delta * 1e3
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
