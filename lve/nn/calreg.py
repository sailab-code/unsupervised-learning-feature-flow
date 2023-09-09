import torch
import torch.nn as nn
import lve


class CALReg(nn.Module):
    """The skeleton of the Cognitive Action Law (CAL) based 'regularization' module.

    It must be extended and customized, still following the interface of this basic class.
    WARNING: all the parameters of a CALReg-layer must be added to the self.all_params list.

    Args:
        reference_layer (nn.Module): layer that must be decorated with CAL-based 'regularization'.
        options (dict): options of this module.
    """
    def __init__(self, reference_layer, options):
        super(CALReg, self).__init__()

        # reference to the layer that will be decorated
        self.reference_layer = [reference_layer]  # using a 1-element list to avoid it being captured as a submodule

        # reference to the options of this module
        self.options = options

        # this flag indicated whether the backward function (see below) should do something or not.
        self.enabled = True

        # flag that determines if bias is used or not
        self.use_bias = False

        # basic parameter placeholders
        self.weight_dot = None

        # append all the layer parameters to this list (weights, biases, ...)
        self.all_params = []

        # stats only parameters
        self.__potential_coherence_global = nn.Parameter(data=torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.__dirty_kinetic_coherence_global = nn.Parameter(data=torch.zeros(1, dtype=torch.float32),
                                                             requires_grad=False)
        self.__custom_coherence_global = nn.Parameter(data=torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.__custom_coherence_moving = nn.Parameter(data=-torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.__coherence_global_n = nn.Parameter(data=torch.zeros(1, dtype=torch.float32), requires_grad=False)

    def forward(self):
        """The function that perform inference-time operations on this module (usually empty or reset-related ops)."""
        raise NotImplementedError("To be implemented!")

    def backward(self, coherence_matrices=None, probabilities_dot=None):
        """This is not a torch-related backward function: it is simply a function sharing the 'backward' name.

        It must be called explicitly after having computed the classic gradients of the loss function.
        It typically performs some computations and it swaps the gradients accordingly to the CAL, so that it is
        safe to call an optimization step in the parameters of this layer (and of the reference layer).

        When implementing the function, please remember to avoid doing anything if self.enabled is false.

        Example:
            if not self.enabled:
            return

        Args:
            coherence_matrices (tuple of tensors): The coherence-related matrices (O,M,N,Md,Nd) needed by CAL.
            probabilities_dot (tuple of tensors): The derivative of the feature probabilities needed by CAL.
        """
        raise NotImplementedError("To be implemented!")

    def compute_lagrangian(self, kinetic, potential, t):
        """The function that computes the value of the Lagrangian (with torch.no_grad():).

        Args:
            kinetic (float): value of the kinetic energy.
            potential (float): value of the potential energy.
            t (int): time index.

        Returns:
            The value of the Lagragian, without any time-scaling factors.
            Time-scaling factor.
        """
        raise NotImplementedError("To be implemented!")

    def compute_kinetic(self):
        """The function that computes the value of the kinetic energy (with torch.no_grad():).

        It might also return other data that could be useful for statistical/debugging purposes.
        """
        raise NotImplementedError("To be implemented!")

    def compute_coherence(self, coherence_matrices):
        """The value of the coherence terms in the Lagrangian.

        Args:
            coherence_matrices (tuple of tensors): The coherence-related matrices (O,M,N,Md,Nd) needed by CAL.

        Returns:
            The value of the coherence-related potential.
            The dirty coherence-related kinetic term.
        """
        self.__disable_grad_wrt_module_params()  # keeping the derivatives out of auto-grad

        # unpacking
        O, M, N, _, _, _ = coherence_matrices  # O, M, N, Md, Nd, each of them vol x vol (or (vol+1) x (vol+1) if bias)

        # shortcuts
        vol = M.shape[1] - (1 if self.use_bias else 0)
        m = self.weight_dot.shape[0]

        Q = self.get_ref_weight().view(m, vol)  # m x vol
        Qd = self.weight_dot.view(m, vol)  # m x vol

        if self.use_bias:
            Q = torch.cat((Q, self.get_ref_bias().view(m, 1)), dim=1)  # m x (vol+1)
            Qd = torch.cat((Qd, self.bias_dot.view(m, 1)), dim=1)  # m x (vol+1)

        qo = torch.matmul(Q, O).view(-1)
        qdm = torch.matmul(Qd, M).view(-1)
        qn = torch.matmul(Q, N).view(-1)

        potential_coherence = torch.dot(qo, Q.view(-1))
        dirty_kinetic_coherence = torch.dot(qdm, Qd.view(-1)) + 2.0 * torch.dot(qn, Qd.view(-1))

        self.__enable_grad_wrt_module_params()  # turning them on again

        return potential_coherence, dirty_kinetic_coherence

    def accumulate_coherence(self, potential_coherence, dirty_kinetic_coherence, custom_coherence=None, fake=False):
        """The value of the coherence terms in the Lagrangian internally accumulated at each call.

        Args:
            potential_coherence: the value of the coherence term in the potential.
            dirty_kinetic_coherence: the value of the coherence terms in the kinetic portion of the Lagrangian.
            custom_coherence: the value of a custom coherence measure.
            fake (boolean flag, optional): avoid updating the accumulated terms.

        Returns:
            The accumulated value of the coherence-related potential.
            The accumulated dirty coherence-related kinetic term.
            The accumulated custom coherence.
        """

        with torch.no_grad():
            if not fake:
                self.__coherence_global_n[0] += 1

                self.__potential_coherence_global[0] += \
                    (potential_coherence - self.__potential_coherence_global[0]) / self.__coherence_global_n[0]
                self.__dirty_kinetic_coherence_global[0] += \
                    (dirty_kinetic_coherence - self.__dirty_kinetic_coherence_global[0]) / self.__coherence_global_n[0]

                if custom_coherence is not None:
                    self.__custom_coherence_global[0] += \
                        (custom_coherence - self.__custom_coherence_global[0]) / self.__coherence_global_n[0]

                    if self.__custom_coherence_moving[0] != -1.0:
                        self.__custom_coherence_moving[0] = \
                            self.__custom_coherence_moving[0] * 0.92 + custom_coherence * 0.08
                    else:
                        self.__custom_coherence_moving[0] = custom_coherence
            else:
                if custom_coherence is not None:
                    if self.__custom_coherence_moving[0] == -1.0:
                        self.__custom_coherence_moving[0] = custom_coherence

            return self.__potential_coherence_global, \
                   self.__dirty_kinetic_coherence_global, \
                   self.__custom_coherence_global if custom_coherence is not None else None, \
                   self.__custom_coherence_moving if custom_coherence is not None else None

    def build_coherence_matrices(self, cur_P, prev_P, delta, prev_M_N=None, coherence_indices=None):
        """The coherence terms in the Lagrangian.

        Args:
            cur_P (tensor): matrix of the patches (receptive fields) of the current frame, b x vol x wh.
            prev_P (tensor): matrix of the patches (receptive fields) of the previous frame, b x vol x wh.
            delta (float): length of the time interval on which derivatives are approximated.
            prev_M_N (tuple of tensors): matrices M and N of the previous frame, each b x vol x vol (def: None).
            coherence_indices (tensor): coords to which each current px is associated in the prev frame (def: None).

        Returns:
            Matrices O,M,N,Md,Nd, where Md,Nd are zeros if no prev_M_N is provided. Matrices are summed-up on the batch.
            The tuple of matrix M and matrix N associated to the last element of the batch are also returned.
        """
        b = cur_P.shape[0] # cur_P is b x vol x wh
        vol = cur_P.shape[1]
        wh = cur_P.shape[2]

        if coherence_indices is not None:
            prev_P = lve.nn.swap_by_indices(prev_P, coherence_indices[0, None])
            if b > 1:
                prev_P_batch = lve.nn.swap_by_indices(cur_P[0:-1,:], coherence_indices[1:,:])
                prev_P = torch.cat((prev_P, prev_P_batch), dim=0)
        else:
            prev_P = torch.cat((prev_P, cur_P[0:-1,:]), dim=0)  # this is when the patches were already reordered

        Gamma = (cur_P - prev_P)  # b x vol x wh
        O_batch = torch.bmm(Gamma, Gamma.transpose(1,2))  # b x vol x vol
        N_batch = torch.bmm(Gamma, cur_P.transpose(1,2))  # b x vol x vol
        M_batch = torch.bmm(cur_P, cur_P.transpose(1,2))  # b x vol x vol

        if self.use_bias:
            _s = torch.sum(cur_P, dim=2).view(b, vol, 1)  # b x vol x 1
            _wh = torch.tensor([wh], dtype=torch.float32, device=_s.device).repeat(b, 1, 1)  # b x 1 x 1
            _s_wh = torch.cat((_s, _wh), dim=1)  # b x (vol+1) x 1
            _zz = torch.zeros_like(_s_wh)  # b x (vol+1) x 1
            _z = torch.zeros_like(_s)  # b x (vol+1) x 1

            M_batch = torch.cat((torch.cat((M_batch, _s), dim=2), _s_wh.view(b, 1, vol+1)), dim=1)
            N_batch = torch.cat((torch.cat((N_batch, _s), dim=2), _zz.view(b, 1, vol+1)), dim=1)
            O_batch = torch.cat((torch.cat((O_batch, _z), dim=2), _zz.view(b, 1, vol+1)), dim=1)

        # absorbing the batch index (and "adapting" the scaling by delta)
        # meaning of "adapting": O should be divided by delta^2, M by 1.0, N by delta
        # i.e., here we assume to multiply the three matrices by delta^2, so the scaling factors are 1, delta^2, delta.
        # this is not an approximation or a weird heuristic, since all these matrices are multiplied by the custom
        # scaling factor lambda_M, and we can assume that lambda_M is scaled instead (however, what we do here is
        # numerically more stable)
        O = torch.sum(O_batch, dim=0) / (wh * b)  # vol x vol
        M = torch.sum(M_batch, dim=0) * (delta * delta) / (wh * b) # vol x vol
        N = torch.sum(N_batch, dim=0) * delta / (wh * b) # vol x vol

        with torch.no_grad():
            if prev_M_N is not None:
                prev_M, prev_N = prev_M_N

                Nd = (N_batch[0, :] - prev_N)  # due to the adaptation above, do not "/ delta"
                if b > 1:
                    Nd_batch = (N_batch[1:,:] - N_batch[0:-1,:])  # due to the adaptation above, do not "/ delta"
                    Nd = Nd.squeeze(0) / (wh * b) + torch.sum(Nd_batch, dim=0) / (wh * b)

                Md = (M_batch[0, :] - prev_M)  # due to the adaptation above, do not "/ delta"
                if b > 1:
                    Md_batch = (M_batch[1:,:] - M_batch[0:-1,:])  # due to the adaptation above, do not "/ delta"
                    Md = Md.squeeze(0) / (wh * b) + torch.sum(Md_batch, dim=0) / (wh * b)
            else:
                Md = torch.zeros_like(M)
                Nd = torch.zeros_like(N)

        return O, M, N, Md, Nd, (M_batch[-1, :], N_batch[-1, :])

    def disable_backward(self):
        """It sets the 'enabled' attribute to False, so that the backward() function should not do anything."""
        self.enabled = False

        self.__disable_grad_wrt_module_params()

    def zero_parameters(self):
        """It sets to zero all the module parameters."""
        if len(self.all_params) == 0:
            raise NotImplementedError("All the layer parameters (weights, biases, ...) must be added to "
                                      "the self.all_params attribute!")
        for param in self.all_params:
            param.zero_()

    def get_ref_layer(self):
        """It returns the reference layer to which this CAL layer refers.

        Returns:
            Reference to the layer to which this CAL layer refers.
        """
        return self.reference_layer[0]

    def get_ref_weight(self):
        """It returns the weight tensor of the reference layer.

        Returns:
            Weight tensor of the reference layer.
        """
        return self.reference_layer[0].weight

    def get_ref_bias(self):
        """It returns the bias tensor of the reference layer.

        Returns:
            Bias tensor of the reference layer.
        """
        return self.reference_layer[0].bias

    def __disable_grad_wrt_module_params(self):
        """It turns off the gradient computations on all the CAL-layer parameters (requires_grad = False)."""
        if len(self.all_params) == 0:
            raise NotImplementedError("All the layer parameters (weights, biases, ...) must be added to "
                                      "the self.all_params attribute!")
        for param in self.all_params:
            param.requires_grad_(False)

    def __enable_grad_wrt_module_params(self):
        """It turns on the gradient computations on all the CAL-layer parameters (requires_grad = True)."""
        if len(self.all_params) == 0:
            raise NotImplementedError("All the layer parameters (weights, biases, ...) must be added to "
                                      "the self.all_params attribute!")
        if self.enabled:
            for param in self.all_params:
                param.requires_grad_(True)
