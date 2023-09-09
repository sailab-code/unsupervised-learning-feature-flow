import torch
import torch.nn as nn
import math

class CALProb(nn.Module):

    def __init__(self, reference_layer, options):
        super(CALProb, self).__init__()

        # shortcut
        self.options = options

        # reference to the layer that will be decorated
        self.reference_layer = [reference_layer]  # using a 1-element list to avoid it being captured as a submodule

        # parameters (variables)
        pm = reference_layer.weight.shape[0]
        self.p_avg = nn.Parameter(data=(torch.ones(pm, dtype=torch.float32) / pm), requires_grad=False)

        if self.options['lambda_s'] > 0.:
            self.s = nn.Parameter(data=(torch.ones(pm, dtype=torch.float32) / pm), requires_grad=True)
            self.s_dot = nn.Parameter(data=torch.zeros(pm, dtype=torch.float32), requires_grad=True)

        # this flag indicates whether the backward function (see below) should do something or not.
        self.enabled = True

        # data used for statistical purposes only
        self.__avg_p_global_squared = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)
        self.__avg_p_global = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)
        self.__avg_p_global_n = nn.Parameter(data=torch.zeros(3, dtype=torch.float32), requires_grad=False)
        self.__avg_p_global_squared_moving = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32),
                                                          requires_grad=False)
        self.__avg_p_global_moving = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)

        self.__avg_plogp_global = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)
        self.__avg_p_global_shannon = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)
        self.__avg_p_global_shannon_n = nn.Parameter(data=torch.zeros(3, dtype=torch.float32), requires_grad=False)
        self.__avg_plogp_global_moving = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32),
                                                      requires_grad=False)
        self.__avg_p_global_shannon_moving = nn.Parameter(data=torch.zeros((3,pm), dtype=torch.float32), requires_grad=False)

    def forward(self):
        """It performs reset operations and it enforces a probabilistic normalization."""
        rt = self.options['reset_thres']

        with torch.no_grad():
            if self.enabled and self.options['lambda_s'] > 0. and rt > 0.:

                # eventually resetting derivatives
                dot_norm = torch.sum(self.s_dot ** 2)

                if dot_norm >= rt:
                    self.s_dot.zero_()

        # enforcing probabilistic normalization
        if self.options['lambda_s'] > 0.:
            mm = torch.min(self.s)
            if mm < 0.:
                ss = self.s - mm + 1.0
            else:
                ss = self.s
            self.s.data = ss / torch.sum(ss)

    def backward(self, probabilities_dot=None):
        """This is not a torch-related backward function: it is simply a function sharing the 'backward' name.

        It must be called explicitly after having computed the classic gradients of the loss function.
        It typically performs some computations and it swaps the gradients accordingly to the CAL, so that it is
        safe to call an optimization step in the parameters of this layer.

        When implementing the function, please remember to avoid doing anything if self.enabled is false.

        Example:
            if not self.enabled:
                return

        Args:
            probabilities_dot (tuple of tensors): derivatives of the feature probabilities needed by CAL.
        """
        if not self.enabled or self.options['lambda_s'] <= 0.:
            return

        with torch.no_grad():
            self.s_dot.grad = - (self.s.grad / (2.0 * self.options['lambda_s'])) - probabilities_dot
            self.s.grad = -self.s_dot

    def compute_mi_approx(self, p, p_avg, weights=None, eval_only=False):
        """The function that computes the value of the (approximated) mutual information.

        If the option 'lambda_s' zero (or smaller than zero) then the CAL-based approximation of the entropy variation
        is not used, and the usual mutual information computation is performed.

        Args:
            p (tensor): tensor of pm pixel-wise probabilities, b x pm x h x w
            p_avg (tensor): tensor of pm average pixel-wise probabilities, b x pm x h x w
            weights (tuple of two float): weight of the conditional entropy and entropy terms, respectively.
            eval_only (boolean): forces the computation of the mutual information of the given probabilities.

        Returns:
            Weighed mutual information.
            Conditional entropy.
            Entropy.
            Value of the equality constraint related to the differential CAL-based entropy.
        """
        cond_entropy = -torch.sum(torch.pow(p, 2)).div(p.shape[0] * p.shape[2] * p.shape[3])
        p_avg = torch.mean(p, dim=(0, 2, 3)) if p_avg is None else p_avg

        if not eval_only and self.options['lambda_s'] > 0.:
            self.s_dot.requires_grad_(False)  # keeping the derivatives out of auto-grad

            # CAL-based entropy
            entropy = -torch.sum(torch.pow(self.s, 2))
            avg_p_constraint_penalty = torch.sum(torch.pow(self.s_dot - p_avg, 2))

            if self.enabled:
                self.s_dot.requires_grad_(True)  # turning them on again
        else:

            # usual entropy
            entropy = -torch.sum(torch.pow(p_avg, 2))
            avg_p_constraint_penalty = torch.tensor(-1.0, device=entropy.device)

        if weights is not None:
            weighed_mi = weights[1] * entropy - weights[0] * cond_entropy
        else:
            weighed_mi = entropy - cond_entropy

        return weighed_mi, cond_entropy, entropy, avg_p_constraint_penalty

    def reset_counters(self):
        self.__avg_p_global_n.data = self.__avg_p_global_n.data * 0.0
        self.__avg_p_global_squared.data = self.__avg_p_global_squared.data * 0.0
        self.__avg_p_global.data = self.__avg_p_global.data * 0.0

        self.__avg_p_global_shannon_n.data = self.__avg_p_global_shannon_n.data * 0.0
        self.__avg_plogp_global.data = self.__avg_plogp_global.data * 0.0
        self.__avg_p_global_shannon.data = self.__avg_p_global_shannon.data * 0.0

    def compute_and_accumulate_global_mi_approx(self, p_list):
        """The function that computes the value of the (approximated) mutual information, accumulating it internally.

        Args:
            p_list (list): list of tensors of the pm pixel-wise probabilities to accumulate, b x pm x h x w

        Returns:
            Accumulated mutual information (exact mean).
            Accumulated mutual information (moving average).
        """

        with torch.no_grad():
            mi = [None] * len(p_list)
            mi_moving = [None] * len(p_list)
            mi_last = [None] * len(p_list)

            for i in range(0, len(p_list)):
                p = p_list[i]

                avg_p_squared = torch.sum(torch.pow(p, 2), dim=(0, 2, 3)).div(p.shape[0] * p.shape[2] * p.shape[3])
                avg_p = torch.mean(p, dim=(0, 2, 3))

                # moving average-related quantities
                if self.__avg_p_global_n[i] > 0:
                    self.__avg_p_global_squared_moving.data[i,:] = self.__avg_p_global_squared_moving.data[i,:] * 0.99 + \
                                                              avg_p_squared * 0.01
                    self.__avg_p_global_moving.data[i,:] = self.__avg_p_global_moving.data[i,:] * 0.99 + avg_p * 0.01
                else:
                    self.__avg_p_global_squared_moving.data[i,:] = avg_p_squared
                    self.__avg_p_global_moving.data[i,:] = avg_p

                cond_entropy_moving = -torch.sum(self.__avg_p_global_squared_moving[i,:])
                entropy_moving = -torch.sum(torch.pow(self.__avg_p_global_moving[i,:], 2))
                mi_moving[i] = entropy_moving - cond_entropy_moving

                # exact mean-related quantities (and updates)
                self.__avg_p_global_n[i] += 1
                self.__avg_p_global_squared.data[i,:] = self.__avg_p_global_squared.data[i,:] + ((avg_p_squared - self.__avg_p_global_squared.data[i,:]) / self.__avg_p_global_n[i])
                self.__avg_p_global.data[i,:] = self.__avg_p_global.data[i,:] + ((avg_p - self.__avg_p_global.data[i,:]) / self.__avg_p_global_n[i])

                cond_entropy = -torch.sum(self.__avg_p_global_squared[i,:])
                entropy = -torch.sum(torch.pow(self.__avg_p_global[i,:], 2))
                mi[i] = entropy - cond_entropy

                cond_entropy = -torch.sum(avg_p_squared)
                entropy = -torch.sum(torch.pow(avg_p, 2))
                mi_last[i] = entropy - cond_entropy

            return mi, mi_moving, mi_last

    def compute_and_accumulate_global_mi_shannon(self, p_list):
        """The function that computes the value of the (Shannon's) mutual information, accumulating it internally.

        Args:
            p_list (list): list of tensors of the pm pixel-wise probabilities to accumulate, b x pm x h x w

        Returns:
            Accumulated mutual information (exact mean).
            Accumulated mutual information (moving average).
        """

        with torch.no_grad():
            mi = [None] * len(p_list)
            mi_moving = [None] * len(p_list)
            mi_last = [None] * len(p_list)

            for i in range(0, len(p_list)):
                p = p_list[i]
                s = math.log(p.shape[1])

                avg_plogp = torch.sum(p * (torch.log(p + 1e-20).div(s)), dim=(0, 2, 3)).div(p.shape[0] * p.shape[2] * p.shape[3])
                avg_p = torch.mean(p, dim=(0, 2, 3))

                # moving average-related quantities
                if self.__avg_p_global_shannon_n[i] > 0:
                    self.__avg_plogp_global_moving.data[i,:] = self.__avg_plogp_global_moving.data[i,:] * 0.99 + \
                                                              avg_plogp * 0.01
                    self.__avg_p_global_shannon_moving.data[i,:] = self.__avg_p_global_shannon_moving.data[i,:] * 0.99 + avg_p * 0.01
                else:
                    self.__avg_plogp_global_moving.data[i,:] = avg_plogp
                    self.__avg_p_global_shannon_moving.data[i,:] = avg_p

                cond_entropy_moving = -torch.sum(self.__avg_plogp_global_moving[i,:])
                entropy_moving = -torch.sum(self.__avg_p_global_shannon_moving[i,:] *
                                            torch.log(self.__avg_p_global_shannon_moving[i,:] + 1e-20).div(s))
                mi_moving[i] = entropy_moving - cond_entropy_moving

                # exact mean-related quantities (and updates)
                self.__avg_p_global_shannon_n[i] += 1
                self.__avg_plogp_global.data[i,:] = self.__avg_plogp_global.data[i,:] + ((avg_plogp - self.__avg_plogp_global[i,:].data) / self.__avg_p_global_shannon_n[i])
                self.__avg_p_global_shannon.data[i,:] = self.__avg_p_global_shannon.data[i,:] + ((avg_p - self.__avg_p_global_shannon.data[i,:]) / self.__avg_p_global_shannon_n[i])

                cond_entropy = -torch.sum(self.__avg_plogp_global[i,:])
                entropy = -torch.sum(self.__avg_p_global_shannon[i,:] *
                                     torch.log(self.__avg_p_global_shannon[i,:] + 1e-20).div(s))
                mi[i] = entropy - cond_entropy

                cond_entropy = -torch.sum(avg_plogp)
                entropy = -torch.sum(avg_p * torch.log(avg_p + 1e-20).div(s))
                mi_last[i] = entropy - cond_entropy

            return mi, mi_moving, mi_last

    def build_prob_vectors(self, cur_probabilities, delta):
        zs = self.options['zeta_s']

        if 0 < zs < 1:
            avg_p = zs * torch.mean(cur_probabilities, dim=(0, 2, 3)) + (1.0 - zs) * self.p_avg.detach()
        else:
            avg_p = torch.mean(cur_probabilities, dim=(0, 2, 3))

        with torch.no_grad():
            p_dot = (avg_p - self.p_avg) / delta
            self.p_avg.data = avg_p

        return p_dot, avg_p

    def disable_backward(self):
        self.enabled = False

        for name, param in self.named_parameters():
            param.requires_grad_(False)
