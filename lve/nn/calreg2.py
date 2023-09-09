import torch
import torch.nn as nn
import lve
from lve.nn.calreg import CALReg


class CALReg2(CALReg):
    def __init__(self, reference_layer, options):
        super(CALReg2, self).__init__(reference_layer, options)

        # module parameters (variables)
        self.weight_dot = None
        self.bias_dot = None

        for name, param in reference_layer.named_parameters():
            if name == "weight":
                self.weight_dot = nn.Parameter(torch.zeros_like(param), requires_grad=True)
            if name == "bias" and param is not None:
                self.bias_dot = nn.Parameter(torch.zeros_like(param), requires_grad=True)

        if self.weight_dot is None:
            raise ValueError("Cannot apply CAL to this layer (no weight tensors were found)")

        # collecting module parameters (variables)
        self.all_params = [self.weight_dot]
        if self.bias_dot is not None:
            self.use_bias = True
            self.all_params.append(self.bias_dot)
        else:
            self.use_bias = False

        # cached computations
        self.__precomputed_norms = None

    def forward(self):
        rt = self.options['reset_thres']

        # eventually resetting derivatives
        with torch.no_grad():
            if self.enabled:
                dot_norm = torch.sum(self.weight_dot ** 2)

                if self.use_bias:
                    dot_norm += torch.sum(self.bias_dot ** 2)

                if 0. < rt <= dot_norm:
                    self.zero_parameters()

                self.__precomputed_norms = [dot_norm]
            else:
                self.__precomputed_norms = None

    def backward(self, coherence_matrices=None, probabilities_dot=None):
        if not self.enabled:
            return

        with torch.no_grad():
            ref_layer = self.get_ref_layer()

            vol = self.weight_dot.shape[1] * self.weight_dot.shape[2] * self.weight_dot.shape[3]
            m = self.weight_dot.shape[0]

            k = self.options['k']
            a = self.options['alpha']
            b = self.options['beta']
            lm = self.options['lambda_m']

            Qd = self.weight_dot.view(m, vol)
            Q = self.get_ref_weight().view(m, vol)

            if self.use_bias:
                Qd = torch.cat((Qd, self.bias_dot.view(m, 1)), dim=1)  # m x (vol+1)
                Q = torch.cat((Q, self.get_ref_bias().view(m, 1)), dim=1)  # m x (vol+1)

            I = torch.eye(vol + (1 if self.use_bias else 0), dtype=torch.float, device=Q.device)

            # changing position of gradient terms, accordingly to CAL ODE
            self.__swap_gradients()

            # getting the potential computed by back-propagation
            if self.weight_dot.grad is not None:
                potential_grad = self.weight_dot.grad.view(m, vol)  # m x vol
                if self.use_bias:
                    potential_grad = torch.cat((potential_grad, self.bias_dot.grad.view(m, 1)), dim=1)  # m x vol+1
            else:
                potential_grad = 0.0

            # checking if we can enforce coherence using CAL matrices
            coher = False
            if isinstance(ref_layer, lve.nn.Conv2dUnf) or isinstance(ref_layer, lve.nn.Conv2dUnf1Px):
                if lm > 0.0 and coherence_matrices is not None:
                    coher = True
                    _, M, N, _, _, _ = coherence_matrices  # O, M, N, Md, Nd, (M,N) last element prev batch
                    Nt = N.t()

            dot_update_term = \
                (torch.matmul(Qd, b * I + ((lm * M) if coher else 0.)) + \
                 (torch.matmul(Q, k * I + (lm * Nt)) if coher else 0.) + \
                 potential_grad) / a  # m x vol (or m x (vol+1) if using bias)

            # updating the gradient terms
            if self.bias_dot is None:
                self.weight_dot.grad = dot_update_term.view(self.weight_dot.shape)
            else:
                self.weight_dot.grad = dot_update_term[:, 0:-1].view(self.weight_dot.shape)
                self.bias_dot.grad = dot_update_term[:, -1].view(self.bias_dot.shape)

    def compute_lagrangian(self, kinetic, potential, t):
        with torch.no_grad():
            return kinetic + potential, 1.0

    def compute_kinetic(self):
        with torch.no_grad():
            b = self.options["beta"]

            weight = self.get_ref_weight()

            if self.__precomputed_norms is None:
                dot_norm = torch.sum(self.weight_dot ** 2)
            else:
                dot_norm = self.__precomputed_norms[0]

            norm = torch.sum(weight ** 2)

            bias = self.get_ref_bias()
            if bias is not None:
                dot_norm += torch.sum(self.bias_dot ** 2)
                norm += torch.sum(bias ** 2)

            dummy = torch.tensor(-1.0, dtype=torch.float32, device=dot_norm.device)

            return (b / 2.0) * dot_norm, dummy, dot_norm, dummy, norm

    def __swap_gradients(self):
        weight = self.get_ref_weight()
        self.weight_dot.grad = weight.grad
        weight.grad = -self.weight_dot

        if self.use_bias:
            bias = self.get_ref_bias()
            self.bias_dot.grad = bias.grad
            bias.grad = -self.bias_dot

