import torch
import torch.nn as nn
import lve
from lve.nn.calreg import CALReg


class CALReg1(CALReg):
    def __init__(self, reference_layer, options):
        super(CALReg1, self).__init__(reference_layer, options)

    def forward(self):
        return

    def backward(self, coherence_matrices=None, probabilities_dot=None):
        if not self.enabled:
            return

        with torch.no_grad():
            weight = self.get_ref_weight()
            bias = self.get_ref_bias()
            k = self.options['k']

            # updating the gradient terms
            weight.grad += k * weight
            if self.use_bias:
                bias.grad += k * bias

    def compute_lagrangian(self, kinetic, potential, t):
        with torch.no_grad():
            return kinetic + potential, 1.0

    def compute_kinetic(self):
        with torch.no_grad():
            k = self.options['k']
            weight = self.get_ref_weight()

            norm = torch.sum(weight ** 2)

            bias = self.get_ref_bias()
            if bias is not None:
                norm += torch.sum(bias ** 2)

        dummy = torch.tensor(-1.0, dtype=torch.float32, device=weight.device)

        return (k / 2.0) * norm, dummy, dummy, dummy, norm


