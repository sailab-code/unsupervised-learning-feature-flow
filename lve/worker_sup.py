import os

import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform, randrange
import lve
import torch
import cv2
import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from collections import OrderedDict

from lve.utils import backward_warp


class UpdatePolicy():
    def __init__(self, options):
        self.length = options['length']
        self.skip = options['skip']

    def is_active(self, i):
        t = i % (self.length + self.skip)
        return t < self.length


def squash(x, gamma, binary=False):
    if binary:
        return (x > gamma).to(x.dtype)
    else:
        return 1 - torch.exp(-gamma * x)


class WorkerSup(lve.Worker):

    def __init__(self, w, h, c, fps, ins, options):
        super().__init__(w, h, c, fps, options)  # do not forget this

        # if the device name ends with 'b' (e.g., 'cpub'), the torch benchmark mode is activated (usually keep it off)
        if options["device"][-1] == 'b':
            self.device = torch.device(options["device"][0:-1])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device(options["device"])
            torch.backends.cudnn.benchmark = False
        self.options = options

        # enforcing a deterministic behaviour, when possible
        # torch.set_deterministic(True)

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # registering supported commands (commands that are triggered by the external visualizer/interface)
        self.register_command("reset_foa", self.__handle_command_reset_foa)
        self.register_command("supervise", self.__handle_command_supervise)

        self.ins = ins
        # saving a shortcut to the neural network options
        self.net_options = self.options["net"]
        self.net_options["w"] = self.w
        self.net_options["h"] = self.h

        # defining processors
        self.optical_flow = lve.OpticalFlowCV(backward=options["backward_optical_flow"])
        self.geymol = lve.GEymol(self.options["foa"], self.device) if self.options["foa"] is not None else None
        self.sup_buffer = lve.SuperWW(device=self.device)
        self.net = lve.NetSup(self.net_options, self.device, self.sup_buffer).to(self.device)

        # neural network optimizer
        self.__lr = self.net_options["step_size"]
        if self.__lr < 0.:  # hack
            self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
        else:
            self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)
        self.__freeze = self.net_options["freeze"]
        if self.__freeze:
            self.net.requires_grad_(False)

        # setting up initial supervision map (if any)
        self.augment_supervision_map(self.options["supervision_map"], self.net_options["supervised_categories"])

        # misc
        self.__ids = torch.arange(self.w * self.h, device=self.device)
        self.__activations_foa_prev = None
        self.__avg_unsupervised_probs = None
        self.__sup_added = False
        self.__sup_new_category_added = False
        self.__frame = None
        self.__old_frames = None
        self.__frame_embeddings = None
        self.__motion_needed = (self.geymol is not None and self.geymol.parameters["alpha_of"] > 0.)
        self.__unsup_loss = torch.tensor(0.).to(self.device)
        self.__sup_loss = torch.tensor(0.).to(self.device)
        self.__stats = OrderedDict(
            [('loss', -1.), ('acc', -1.), ('f1', -1.), ('sup_loss', -1.), ('spatial_coherence', -1.),
             ('closeness', -1.)]
        )  # placeholders
        self.__foa = None
        self.eps = 0.000000001
        self.define_derivatives()

        # self.counter_sup = 0

    def compute_features_motion_closeness_loss(self, d_feat, d_flow, loss_type="standard"):
        closeness_terms = []
        closeness_2d = []
        closeness_w_terms = []
        motion_mask_percentage_list = []
        for i in range(len(d_feat)):
            feat_dx, feat_dy = d_feat[i]  # [b, c, h, w ]
            flow_dx, flow_dy = d_flow[i]  # [b, 2, h, w]
            flow_dx, flow_dy = flow_dx.detach(), flow_dy.detach()

            # flow_norm = 0.5 * torch.sum((flow_dx ** 2 + flow_dy ** 2), dim=1, keepdim=True)
            flow_norm_x = 0.5 * torch.sum(flow_dx ** 2, dim=1, keepdim=True)
            flow_norm_y = 0.5 * torch.sum(flow_dy ** 2, dim=1, keepdim=True)

            # feature_norm = (1 / feat_dx.size(1)) * torch.sum((feat_dx ** 2 + feat_dy ** 2), dim=1, keepdim=True)
            feature_norm_x = (1 / feat_dx.size(1)) * torch.sum(feat_dx ** 2, dim=1, keepdim=True)
            feature_norm_y = (1 / feat_dy.size(1)) * torch.sum(feat_dy ** 2, dim=1, keepdim=True)

            if loss_type == "standard":
                # features_component = flow_norm / (feature_norm + self.eps)  # TODO add weights where they should be
                features_component = flow_norm_x / (feature_norm_x + self.eps) + flow_norm_y / (
                        feature_norm_y + self.eps)  # TODO add weights where they should be
            elif loss_type == "logaritmic":
                # features_component = torch.log(self.eps + flow_norm) - torch.log(self.eps + feature_norm)
                features_component = (torch.log(self.eps + flow_norm_x) - torch.log(self.eps + feature_norm_x)) + \
                                     (torch.log(self.eps + flow_norm_y) - torch.log(self.eps + feature_norm_y))
            elif "squash" in loss_type:
                binary_flag = loss_type == "bsquash"
                g_dm = self.net_options['vision_block']['gamma_dm']
                g_dphi = self.net_options['vision_block']['gamma_dphi']
                features_component = squash(flow_norm_x, gamma=g_dm, binary=binary_flag) * (
                            1 - squash(feature_norm_x, gamma=g_dphi, binary=binary_flag)) + squash(flow_norm_y, gamma=g_dm, binary=binary_flag) * (
                                                 1 - squash(feature_norm_y, gamma=g_dphi, binary=binary_flag))
            else:
                raise NotImplementedError
            closeness_2d.append(features_component)
            # # motion_component = feature_norm * 1 / (flow_norm + self.eps)
            # loss = torch.mean(
            #     motion_mask * (motion_component * self.options['vision_block']['features']['lambda_mf1'][i]
            #                    + features_component * self.options['vision_block']['features']['lambda_fm'][i]))
            # loss = torch.mean(features_component * self.options['vision_block']['features']['lambda_fm'][i])
            loss = torch.mean(features_component)
            closeness_terms.append(loss)
            closeness_w_terms.append(loss * self.net_options['vision_block']['features']['lambda_mf'][i])

        return {'mf_closeness': torch.mean(torch.stack(closeness_terms)),
                'mf_closeness_2d': closeness_2d,
                'mf_closeness_w': torch.mean(
                    torch.stack(closeness_w_terms))}, motion_mask_percentage_list

    def compute_features_spatial_coherence_loss(self, d_feat, d_flow, flow, loss_type='standard'):
        features_coherence_terms = []
        features_coherence_w_terms = []
        edge_aware_mask_list_x = []
        edge_aware_mask_list_y = []
        fsmoothness_2d = []
        for i in range(self.net_options['n_blocks']):
            if d_flow is not None and (loss_type == 'standard' or loss_type =='logaritmic'):
                m_x, m_y = d_flow[i]
                m_x, m_y = m_x.detach(), m_y.detach()  # detach the features!
                exp_x = torch.exp(
                    - self.net_options['vision_block']['features']['lambda_e'][i] * torch.mean(m_x ** 2, dim=1))
                exp_y = torch.exp(
                    - self.net_options['vision_block']['features']['lambda_e'][i] * torch.mean(m_y ** 2, dim=1))
                edge_aware_mask_list_x.append(exp_x.detach())
                edge_aware_mask_list_y.append(exp_y.detach())
                feat_dx, feat_dy = d_feat[i]
                feat_dx = feat_dx ** 2 * exp_x.unsqueeze(dim=1)  # TODO check if broadcasting works as we want
                feat_dy = feat_dy ** 2 * exp_y.unsqueeze(dim=1)
                penalty = feat_dx + feat_dy
            elif "squash" in loss_type:
                binary_flag = loss_type == "bsquash"
                m_x, m_y = d_flow[i]
                flow_dx, flow_dy = m_x.detach(), m_y.detach()  # detach the
                feat_dx, feat_dy = d_feat[i]

                flow_dx = 0.5 * torch.sum(flow_dx ** 2, dim=1, keepdim=True)
                flow_dy = 0.5 * torch.sum(flow_dy ** 2, dim=1, keepdim=True)
                feat_dx = (1 / feat_dx.size(1)) * torch.sum(feat_dx ** 2, dim=1, keepdim=True)
                feat_dy = (1 / feat_dy.size(1)) * torch.sum(feat_dy ** 2, dim=1, keepdim=True)

                g_m = self.net_options['vision_block']['gamma_m']
                g_dm = self.net_options['vision_block']['gamma_dm']
                g_dphi = self.net_options['vision_block']['gamma_dphi']
                m = torch.sum((flow[i] ** 2).detach(), dim=1, keepdim=True)
                penalty = squash(m, gamma=g_m, binary=binary_flag) * (
                        (1 - squash(flow_dx, gamma=g_dm, binary=binary_flag)) * squash(feat_dx, gamma=g_dphi, binary=binary_flag) +
                        (1 - squash(flow_dy, gamma=g_dm, binary=binary_flag)) * squash(feat_dy, gamma=g_dphi, binary=binary_flag)
                )
            else:
                raise NotImplementedError

            fsmoothness_2d.append(torch.mean(penalty, dim=1, keepdim=True))
            feature_norm = 0.5 * torch.mean(penalty)
            features_coherence_terms.append(feature_norm)
            features_coherence_w_terms.append(feature_norm *
                                              self.net_options['vision_block']['features']['lambda_sp'][i])

        return {'spatial_coherence': torch.mean(torch.stack(features_coherence_terms)),
                'fsmoothness_2d': fsmoothness_2d,
                'spatial_coherence_w': torch.mean(
                    torch.stack(features_coherence_w_terms))}, edge_aware_mask_list_x, edge_aware_mask_list_y

    def define_derivatives(self):
        self.sobel_dx_kernel = torch.Tensor([[1 / 2, 0, -1 / 2],
                                             [1, 0, -1],
                                             [1 / 2, 0, -1 / 2]]).to(self.device)
        self.sobel_dy_kernel = torch.Tensor([[1 / 2, 1, 1 / 2],
                                             [0, 0, 0],
                                             [-1 / 2, -1, -1 / 2]]).to(self.device)

        self.hs_dx_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, 1 / 4],
                                          [0, -1 / 4, 1 / 4]]).to(self.device)

        self.hs_dy_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, -1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(self.device)

        self.hs_dt_kernel = torch.Tensor([[0, 0, 0],
                                          [0, 1 / 4, 1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(self.device)

        self.hs_filter = torch.Tensor([[1 / 12, 1 / 6, 1 / 12],
                                       [1 / 6, 0, 1 / 6],
                                       [1 / 12, 1 / 6, 1 / 12]]).to(self.device)

        if self.net_options['c'] == 3:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        elif self.net_options['c'] == 1:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3))
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3))
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3))
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3))
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3))
        else:
            raise NotImplementedError

        self.hs_dx_kernel_uv = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.hs_dy_kernel_uv = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_x_uv = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_y_uv = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)

    def feature_gradient(self, features):

        channels = features.shape[1]
        grad_x_weights = self.sobel_dx_kernel
        grad_y_weights = self.sobel_dy_kernel

        grad_x_weights = grad_x_weights.expand(channels, 1, 3, 3)
        grad_y_weights = grad_y_weights.expand(channels, 1, 3, 3)

        grad_x = torch.nn.functional.conv2d(features, grad_x_weights, groups=features.shape[1], padding=1)
        grad_y = torch.nn.functional.conv2d(features, grad_y_weights, groups=features.shape[1], padding=1)
        return grad_x, grad_y

    def gradient(self, x, x_=None, type="sobel"):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        batch_dims = x.shape[0]
        channel_dims = x.shape[1]

        if type == "shift":
            left = x
            right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            dx, dy = right - left, bottom - top
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

        elif type == "sobel":
            if channel_dims == 3 or channel_dims == 1:  # on frames
                dx = torch.nn.functional.conv2d(x, self.sobel_dx_kernel_f, stride=1, padding=1,
                                                groups=channel_dims)  # TODO check is poosible padding="same"
                dy = torch.nn.functional.conv2d(x, self.sobel_dy_kernel_f, stride=1, padding=1, groups=channel_dims)
            elif channel_dims == 2:  # on flow
                dx = torch.nn.functional.conv2d(x, self.sobel_dx_kernel_uv, stride=1, padding=1, groups=channel_dims)
                dy = torch.nn.functional.conv2d(x, self.sobel_dy_kernel_uv, stride=1, padding=1, groups=channel_dims)
            else:
                raise NotImplementedError

        elif type == "hs":
            if channel_dims == 3 or channel_dims == 1:  # on frames
                a = self.hs_dx_kernel_f
                b = self.hs_dy_kernel_f
                dx = F.conv2d(x, a, stride=1, padding=1, groups=channel_dims) + \
                     F.conv2d(x_, a, stride=1, padding=1, groups=channel_dims)
                dy = F.conv2d(x, b, stride=1, padding=1, groups=channel_dims) + \
                     F.conv2d(x_, b, stride=1, padding=1, groups=channel_dims)
            elif channel_dims == 2:  # on flow
                a = self.hs_dx_kernel_uv
                b = self.hs_dy_kernel_uv
                dx = F.conv2d(x, a, stride=1, padding=1, groups=channel_dims) * 2
                dy = F.conv2d(x, b, stride=1, padding=1, groups=channel_dims) * 2
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return dx, dy

    def compute_gradients_list(self, tensor, type="features"):
        list_grad = []
        for i in range(self.net_options['n_blocks']):
            if type == "features":
                tensor_dx, tensor_dy = self.feature_gradient(tensor[i])
            elif type == "displacements":
                tensor_dx, tensor_dy = self.gradient(tensor[i], type=self.options['gradient_type'])
            else:
                raise NotImplementedError
            list_grad.append([tensor_dx, tensor_dy])
        return list_grad

    def process_frame(self, frame, of=None, supervisions=None, unfiltered_supervisions=None, foa=None):

        # data returned by the call below and their types are:
        # frame: input frame (torch tensor, 1 x c x h x w, better keep the dummy batch dimension here),
        # motion: optical flow (torch tensor, 1 x 2 x h x w, better keep the dummy batch dimension here),
        # foa_row_col: focus of attention - integer coordinates (torch tensor, 2 elements)
        # saccade_bool: flag that tell if the movement of the focus of attention is a saccade (bool)
        # foa: focus of attention - real x, y, velocity_x, velocity_y (list of 4 elements)
        # sups: a pair of torch tensors (targets, indices), or (None, None)

        frame, motion, foa_row_col, saccade_bool, foa, sup_targets, sup_indices, sup_targets_unf, sup_indices_unf = \
            self.__compute_missing_data_and_convert(frame, of, foa, supervisions, unfiltered_supervisions)
        if self.__old_frames is None:
            self.__old_frames = frame

        # we override saccades with a flag that is only about movement
        saccade_bool = torch.norm(motion[0, :, foa_row_col[0], foa_row_col[1]]) == 0

        if self.net_options["normalize"]:
            # normalization (customizable)
            frame = (frame - 0.5) / 0.25

        # sampling a/some buffered frame(s) (associated to previous supervisions - if any)
        buff_frames = self.sup_buffer.sample_frames(self.options['piggyback_frames'],
                                                    self.options['sup_persistence'])
        buff_frames = buff_frames.to(self.device) if buff_frames is not None else None

        # standard one-frame online mode
        old_frames_batch = self.__old_frames
        frame_batch = frame

        # inference (full inference on the current frame, encoding only on the buffered frames - if any)
        #   the current frame and the buffered frames constitute a batch of frames (index 0: current frame)
        # notice: features_current, displacements are lists
        features_current, features_old, _, _, format_required_features, format_required_piggyback, \
        displacements, supervised_probs_masked_list, prediction_mask_list, prediction_idx_list, unmasked_prediction_idx, format_required_old_features = \
            self.net(frame_batch, old_frames_batch, buff_frames, motion)

        self.__frame_embeddings = format_required_features
        piggyback_frame_embeddings = format_required_piggyback

        # updating embeddings of the supervised pixel(s) of the buffered frames (if any)
        self.sup_buffer.update_embeddings_of_sampled_frames(piggyback_frame_embeddings.detach()
                                                            if piggyback_frame_embeddings.shape[0] > 0 else None)

        # storing supervisions associated to the current frame, provided by the input stream (if any)
        self.__sup_added = self.sup_buffer.add(frame, sup_targets, sup_indices, self.__frame_embeddings)

        # updating the worker-level information about supervision (worker has its own counts, exposed to the visualizer)
        self.__update_internal_supervision_counts()

        # computing the motion-based 'blob' around the focus of attention
        foa_blob = self.__compute_foa_moving_blob(motion, foa_row_col,
                                                  motion_threshold=self.options['motion_threshold'])

        # getting the activation associated to the focus-of-attention-coordinates at every layer
        activations_foa = self.__frame_embeddings[None, 0, :, foa_row_col[0], foa_row_col[1],
                          None]

        d_feat = None
        d_flow = None
        if sum(self.net_options['vision_block']['features']['lambda_sp']) > 0.:
            # compute here gradients, only once
            d_feat = self.compute_gradients_list(features_current)
            d_flow = self.compute_gradients_list(displacements['fwd'])

            spatial_coherence_args = {'d_feat': d_feat, 'd_flow': d_flow, 'flow': displacements['fwd'],
                                      'loss_type': self.net_options['regularization_type']}
            feature_spatial_coherence_dic, edge_aware_fmask_list_x, edge_aware_fmask_list_y = self.compute_features_spatial_coherence_loss(
                **spatial_coherence_args)
            spatial_coherence_w_loss = feature_spatial_coherence_dic['spatial_coherence_w']
            spatial_coherence_loss = feature_spatial_coherence_dic['spatial_coherence']
        else:
            spatial_coherence_w_loss, spatial_coherence_loss = torch.zeros(1).to(frame.device), torch.zeros(1).to(
                frame.device)
            feature_spatial_coherence_dic = {"fsmoothness_2d": None}

        # compute everytime - in order to have a sort of closeness metric
        if sum(self.net_options['vision_block']['features']['lambda_mf']) > 0.:

            d_feat = self.compute_gradients_list(features_current)
            d_flow = self.compute_gradients_list(displacements['fwd'])

            # notice: features computed at this block and velocities (that are computed from the input features of the block)
            mf_loss_dic, motion_mask_percentage_list = self.compute_features_motion_closeness_loss(d_feat,
                                                                                                   d_flow,
                                                                                                   loss_type=
                                                                                                   self.net_options[
                                                                                                       "regularization_type"])
        else:
            mf_loss_dic = {'mf_closeness_w': torch.zeros(1).to(frame.device),
                           'mf_closeness_2d': [torch.zeros(
                               (1, 1, self.net_options["h"], self.net_options["w"]))] * len(
                               features_current),
                           'mf_closeness': torch.zeros(1).to(frame.device)}

        # TODO check this batch hack
        if sup_targets_unf is not None:

            # if received all the unfiltered supervisions, train
            flattened_features = format_required_features.flatten(start_dim=2)[0].permute(1, 0)
            sup_loss, sup_details, accuracy, per_class_f1, global_f1, max_idx_class = self.net.compute_supervised_loss(
                flattened_features, sup_targets_unf.to(self.device))

            total_loss = sup_loss  # + spatial_coherence_w_loss + mf_loss_dic["mf_closeness_w"]
        else:
            total_loss, sup_details, accuracy, per_class_f1, global_f1, max_idx_class = torch.zeros(1).to(
                self.device), 0.0, torch.tensor(
                -1).to(self.device), -torch.ones(size=(self.net_options["supervised_categories"],)).to(
                self.device), torch.tensor(-1).to(self.device), None
            # spatial_coherence_w_loss, spatial_coherence_loss = torch.zeros(1).to(frame.device), torch.zeros(1).to(
            #     frame.device)
            # mf_loss_dic = {'mf_closeness_w': torch.zeros(1).to(frame.device),
            #                'mf_closeness_2d': [torch.zeros((1, 1, self.net_options["h"], self.net_options["w"]))] * len(
            #                    features_current),
            #                'mf_closeness': torch.zeros(1).to(frame.device)}
            # feature_spatial_coherence_dic = {"fsmoothness_2d": None}

        # saving output data related to the current frame
        foa_blob = lve.utils.torch_float_01_to_np_uint8(foa_blob.to(torch.float))

        supervised_probs_masked_list = [el.detach().t(). \
                                            view(1, el.shape[1], self.h, self.w).cpu().numpy() for el in
                                        supervised_probs_masked_list]

        prediction_idx_list_detached = [el.detach().cpu().numpy() for el in prediction_idx_list]

        prediction_mask = lve.utils.torch_float_01_to_np_uint8(
            prediction_mask_list[0].view(self.h, self.w).to(torch.float))
        sup_targets = self.sup_buffer.get_last_frame_targets().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None
        sup_indices = self.sup_buffer.get_last_frame_indices().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None

        self.__stats.update({"loss": total_loss.item(),
                             "acc": accuracy.item(),
                             "f1": global_f1.item(),
                             "sup_loss": sup_details,
                             "spatial_coherence": spatial_coherence_loss.item(),
                             "closeness": mf_loss_dic["mf_closeness"].item(),
                             "foax": foa[0], "foay": foa[1],
                             "foavx": foa[2], "foavy": foa[3],
                             "saccade": int(saccade_bool),
                             # "current_threshold": self.net_options["net"]["dist_threshold"]
                             }
                            )

        features_numpy = self.__frame_embeddings.detach().cpu().numpy()
        old_features_numpy = format_required_old_features.detach().cpu().numpy()
        warped_features = backward_warp(frame=self.__frame_embeddings, displacement=displacements['fwd'][-1][0, None])
        warped_features = warped_features.detach().cpu().numpy()
        sup_prediction_idx = max_idx_class.detach().cpu().numpy() if max_idx_class is not None else None
        self.add_outputs({"motion": of[0],  # bin
                          "per_class_f1": per_class_f1.detach().cpu().numpy(),
                          "features": features_numpy,  # bin
                          "old_features": old_features_numpy,  # bin
                          "warped_features": warped_features,  # bin
                          "old_frames": self.__old_frames.detach().cpu().numpy().transpose(0, 2, 3, 1),  # bin
                          "blob": foa_blob,  # image
                          "unmasked-prediction_idx": unmasked_prediction_idx.detach().cpu().numpy(),
                          "sup-probs": supervised_probs_masked_list[0],  # bin
                          "sup-probs-list": supervised_probs_masked_list,  # bin
                          "prediction_idx": prediction_idx_list_detached[0],
                          "sup_prediction_idx": sup_prediction_idx,
                          "prediction_idx-list": prediction_idx_list_detached,
                          "pred-mask": prediction_mask,  # bin
                          "sup.targets": sup_targets,  # bin
                          "sup.indices": sup_indices,  # bin
                          "sup.map": self.get_supervision_map() if self.__sup_new_category_added else None,  # JSON
                          "stats.worker": self.__stats,  # dictionary
                          "logs.worker": list(self.__stats.values()),  # CSV log
                          "tb.worker": self.__stats}, batch_index=0)  # tensorboard

        # network predicted motions
        l = {'closeness_2d': mf_loss_dic['mf_closeness_2d'],
             'fsmoothness_2d': feature_spatial_coherence_dic["fsmoothness_2d"]}
        for i in range(0, len(displacements['fwd'])):
            self.add_output("net_motion." + str(i),
                            displacements['fwd'][i].detach().cpu().numpy())  # network predicted motions
            if l['closeness_2d'] is not None and l['closeness_2d'][i] is not None:
                self.add_output("closeness_2d." + str(i),
                                l['closeness_2d'][i].detach().cpu().numpy())
            if l['fsmoothness_2d'] is not None and l['fsmoothness_2d'][i] is not None:
                self.add_output("fsmoothness_2d." + str(i),
                                l['fsmoothness_2d'][i].detach().cpu().numpy())

        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        self.__activations_foa_prev = activations_foa.detach()

        self.__sup_new_category_added = False
        self.__frame = frame
        self.__old_frames = frame
        self.__foa = (foa[0], foa[1])
        self.__sup_loss = total_loss

    def update_model_parameters(self):

        # check if the freeze option was changed while the agent was running (and react to such change)
        if self.__freeze != self.net_options['freeze']:
            self.__freeze = self.net_options['freeze']
            if self.__freeze:
                self.net.requires_grad_(False)
            else:
                self.net.requires_grad_(True)  # we still have not computed gradients, better 'return'
            return

        # if frozen, nothing to do here
        if self.__freeze:
            return

        # completing the loss function
        loss = self.__unsup_loss + self.__sup_loss

        if loss > 0:
            loss.backward()
            # update step
            self.net_optimizer.step()
            self.net.zero_grad()

        # detaching last frame supervisions (if any)
        if self.__sup_added:
            self.sup_buffer.detach_last_frame_supervisions()

        # check if learning rate was changed (hot)
        if self.__lr != self.net_options['step_size']:
            self.__lr = self.net_options['step_size']
            if self.__lr < 0.:
                self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
            else:
                self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth", map_location=self.device))

        # loading worker-status related tensors
        worker_status = torch.load(worker_model_folder + "worker.pth", map_location=self.device)
        self.__activations_foa_prev = worker_status["what_for_prev"]
        self.__avg_unsupervised_probs = worker_status["avg_unsupervised_probs"]
        self.sup_buffer = worker_status["supervision_buffer"]

        # loading worker-status parameters from a readable JSON
        params = lve.utils.load_json(worker_model_folder + "worker.json")

        # setting up the internal elements using the JSON-loaded parameters
        if self.geymol is not None:
            self.geymol.reset(params["foa_y"], params["foa_t"])
            self.geymol.first_call = False
            self.geymol.IOR_matrix = worker_status["foa_ior_matrix"]

        self.augment_supervision_map(params["supervision_map"],
                                     self.net_options["supervised_categories"],
                                     counts=params["supervision_count"])

        if self.options["supervision_map"] is not None and len(self.options["supervision_map"]) > 0:
            print("WARNING: the provided supervision map will be overwritten by the one loaded from disk!")

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving neural network weights
        torch.save(self.net.state_dict(), worker_model_folder + "net.pth")

        # saving worker-status related tensors
        torch.save({"what_for_prev": self.__activations_foa_prev,
                    "avg_unsupervised_probs": self.__avg_unsupervised_probs,
                    "supervision_buffer": self.sup_buffer,
                    "foa_ior_matrix": self.geymol.IOR_matrix if self.geymol is not None else torch.zeros(
                        (self.h, self.w), dtype=torch.float32, device=self.device)},
                   worker_model_folder + "worker.pth")

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "worker.json",
                            {"foa_y": list(self.geymol.y) if self.geymol is not None else [0., 0., 0., 0.],
                             "foa_t": self.geymol.t if self.geymol is not None else 0,
                             "supervision_map": self.get_supervision_map(),
                             "supervision_count": self.get_supervision_count()})

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "motion": {'data_type': lve.OutputType.MOTION, 'per_frame': True},
            "per_class_f1": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "warped_features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "old_features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "old_frames": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "blob": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sampled-points": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "unsup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "unsup-probs_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup_prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "unmasked-prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "prediction_idx-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "pred-mask": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sup.indices": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.targets": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.map": {'data_type': lve.OutputType.JSON, 'per_frame': False},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys())  # first line of CSV
        }

        for i in range(0, self.options['net']['n_blocks']):
            output_types.update({"net_motion." + str(i): {'data_type': lve.OutputType.MOTION, 'per_frame': True}})
            output_types.update({"fsmoothness_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"closeness_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})

        # output_types.update({"predicted_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True}})

        return output_types

    def print_info(self):
        s = "   worker {"
        i = 0
        for k, v in self.__stats.items():
            s += (k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
            if (i + 1) % 7 == 0:
                if i < len(self.__stats) - 1:
                    s += ",\n           "
                else:
                    s += "}"
            else:
                if i < len(self.__stats) - 1:
                    s += ", "
                else:
                    s += "}"
            i += 1

        print(s)

    def __handle_command_reset_foa(self, command_value, batch_index=0):
        if batch_index:
            pass
        if self.geymol is not None:
            self.geymol.reset([command_value['y'], command_value['x'],
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1)),
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))])
            return True
        else:
            return False

    def __handle_command_supervise(self, command_value, batch_index=0):

        # here is the thing: this function accepts supervision given on a bounding box,
        # where (row, col) is the upper-left coordinate of the box of size (n_rows, n_cols).
        # of course, when a single pixel is supervised, we have (n_rows = 1, n_cols = 1)
        class_name = command_value['class'].lower().strip()
        row = command_value['y']
        col = command_value['x']
        n_cols = command_value['w']
        n_rows = command_value['h']
        n_sup = n_cols * n_rows  # total number of supervised pixels

        # from class name to numerical index (it will also create a new category - for never-seen-before class names)
        target, new_class_added = self.get_target(class_name, self.net_options["supervised_categories"])

        # checking if the system was unable to add any more classes (maximum number of categories has been reached)
        if target is None:
            return False

        # marking the flag that tells that a new category was added (then we will save the updated class map to disk)
        self.__sup_new_category_added = self.__sup_new_category_added or new_class_added

        # preparing supervision data: targets, indices
        # here we have to get the indices of the pixels in the supervised box
        # boxes are expected to be relatively small, so this procedure will be fast
        if n_sup > 1:
            list_of_indices_on_each_row = [None] * n_rows
            for r in range(0, n_rows):
                list_of_indices_on_each_row[r] = torch.arange(n_cols, dtype=torch.long) + (row + r) * self.w + col
            indices = torch.cat(list_of_indices_on_each_row)  # CPU
            targets = torch.ones(n_sup, dtype=torch.long) * target  # CPU
        else:
            indices = torch.ones(1, dtype=torch.long) * row * self.w + col  # CPU
            targets = torch.ones(1, dtype=torch.long) * target  # CPU

        # flag to mark whether this frame has already received supervisions (from input stream of from the visualizer)
        frame_was_already_supervised_before = self.__sup_added

        # buffering (and merging/intersecting) supervisions (warning: supervisions can be overwritten)
        self.__sup_added = self.sup_buffer.add(self.__frame, targets, indices, self.__frame_embeddings,
                                               frame_was_already_added=frame_was_already_supervised_before)

        # saving/updating internal worker stats (after having buffered/intersected them)
        self.__update_internal_supervision_counts(frame_was_already_considered=frame_was_already_supervised_before)

        # sampling or re-sampling (it the batch was already sampled in other methods on in another call to this method)
        embeddings, labels = self.sup_buffer.sample_embeddings_batch(self.options['sup_batch'],
                                                                     self.options['sup_persistence'])
        embeddings = embeddings.to(self.device) if embeddings is not None else None
        labels = labels.to(self.device) if labels is not None else None

        # loss (supervised)
        self.__sup_loss, sup_details = self.net.compute_supervised_loss(embeddings, labels)

        # adding/updating supervision-related output
        sup_targets = self.sup_buffer.get_last_frame_targets().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None
        sup_indices = self.sup_buffer.get_last_frame_indices().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None

        self.__stats.update({"loss": self.__unsup_loss.item() + self.__sup_loss.item(),
                             "loss_sup": sup_details})

        self.add_outputs({"sup.targets": sup_targets,  # bin
                          "sup.indices": sup_indices,  # bin
                          "sup.map": self.get_supervision_map() if self.__sup_new_category_added else None,  # JSON
                          "stats.worker": self.__stats})  # dictionary

    def __microsaccade_crop(self, img, microsaccades, removed_percentage=0.05):

        microsaccade_batch = []
        for j in range(microsaccades):
            max_pixel_removed = int(removed_percentage * min(img.shape[2], img.shape[3]))
            a, b = torch.randint(0, max_pixel_removed, (2,), device=self.device)
            a, b = a - max_pixel_removed // 2, b - max_pixel_removed // 2  # (a columns, b rows)
            top, left, height, width = 0, 0, img.shape[2], img.shape[3]

            if b > 0:
                top = top + b
                height = height - b  # TODO check! - to avoid picking outside
            else:
                height = height + b  # b is negative
            if a > 0:
                left = left + a
                width = width - a  # TODO check! - to avoid picking outside
            else:
                width = width + a  # a is negative
            microsaccade_batch.append(
                transforms.functional.resized_crop(img, top=top, left=left, height=height, width=width,
                                                   size=img.shape[2:]))

        return torch.cat(microsaccade_batch, dim=0)

    def __compute_missing_data_and_convert(self, batch_frames_np_uint8, batch_motion_np_float32,
                                           batch_foa_np_float32, batch_sup_np, batch_unfiltered_sup_np):

        # assumption: data are stored in batches of size 1, i.e., one frame at each time instant
        assert len(batch_frames_np_uint8) == 1, "We are assuming to deal with batches of size 1, " \
                                                "and it does not seem to be the case here!"

        # convert to tensor
        frame_np_uint8 = batch_frames_np_uint8[0]
        frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8, device=self.device)

        # grayscale-instance of the input frame
        if not self.frame_is_gray_scale:
            frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
            frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8, device=self.device)
        else:
            frame_gray_np_uint8 = frame_np_uint8
            frame_gray = frame

        # optical flow
        if batch_motion_np_float32 is None or batch_motion_np_float32[0] is None:
            if self.__motion_needed:
                motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
            else:
                motion_np_float32 = np.zeros((self.h, self.w, 2), dtype=np.float32)

            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

            if batch_motion_np_float32 is not None:
                batch_motion_np_float32[0] = motion_np_float32  # updating, it might be used out of this function
        else:
            motion_np_float32 = batch_motion_np_float32[0]  # h x w x 2

            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

        # focus of attention (a negative dissipation triggers a random trajectory, useful in some experiments)
        if batch_foa_np_float32 is None or batch_foa_np_float32[0] is None:
            if self.geymol is not None:
                if self.options["foa"]["dissipation"] >= 0.0:
                    foa, saccade_bool = self.geymol.next_location(frame_gray, motion,
                                                                  frame_gray_uint8_cpu=frame_gray_np_uint8)
                else:
                    foa = np.array([randrange(0, self.h - 1), randrange(0, self.w - 1), 0., 0.], dtype=np.float64)
                    saccade_bool = False
            else:
                foa = np.array([0., 0., 0., 0.], dtype=np.float64)
                saccade_bool = False
        else:
            foa = batch_foa_np_float32[0][0:4]
            saccade_bool = bool(batch_foa_np_float32[0][-1])

        # getting the two integer coordinates of the focus of attention (discarding velocity)
        foa_row_col = torch.from_numpy(foa[0:2].astype(np.long)).to(self.device).view(2).to(torch.long)
        # a pair of (targets, indices)
        if batch_sup_np is not None and batch_sup_np[0] is not None:
            sup_targets, sup_indices = torch.tensor(batch_sup_np[0][0], dtype=torch.int64), torch.tensor(
                batch_sup_np[0][1])  # CPU
        else:
            sup_targets, sup_indices = None, None

        if batch_unfiltered_sup_np is not None and batch_unfiltered_sup_np[0] is not None:
            sup_targets_unf, sup_indices_unf = torch.tensor(batch_unfiltered_sup_np[0][0],
                                                            dtype=torch.int64), torch.tensor(
                batch_unfiltered_sup_np[0][1])  # CPU
        else:
            sup_targets_unf, sup_indices_unf = None, None

        return frame, motion, foa_row_col, saccade_bool, foa, sup_targets, sup_indices, sup_targets_unf, sup_indices_unf

    def __compute_foa_moving_blob(self, motion, foa_row_col, motion_threshold=0.1):

        if motion_threshold < 0:
            moving_bool = torch.sum(motion ** 2, dim=1).view(self.h, self.w) > 0
            if moving_bool[foa_row_col[0], foa_row_col[1]]:
                return moving_bool
            else:
                return torch.zeros((self.h, self.w), dtype=torch.bool, device=motion.device)

        # boolean mask of those pixels that are moving
        moving_bool = torch.sum(motion ** 2, dim=1).view(-1) > (motion_threshold * motion_threshold)

        # boolean mask (flat) that, at the end of this function, will indicate the pixels belonging to the blob
        blob_bool = torch.zeros(self.h * self.w, dtype=torch.bool, device=motion.device)

        if not moving_bool[foa_row_col[0] * self.w + foa_row_col[1]]:
            return blob_bool.view(self.h, self.w)

        # first of all, we mark the current position of the focus of attention as belonging to the blob
        frontier = foa_row_col[0] * self.w + foa_row_col[1]
        blob_bool[frontier] = True

        while torch.numel(frontier) > 0:
            # expanding the frontier
            frontier_neighs = self.__get_neighbors(frontier).view(-1)

            # removing repeated elements
            frontier_neighs = torch.unique(frontier_neighs)

            # keeping only new elements that are moving
            frontier = torch.masked_select(frontier_neighs,
                                           (blob_bool[frontier_neighs] == 0) * (moving_bool[frontier_neighs] != 0))

            # expanding the blob
            blob_bool[frontier] = True

        # reshaping
        return blob_bool.view(self.h, self.w)

    def __get_neighbors(self, _pixel_ids):
        _w = self.w
        _h = self.h

        # boolean masks of pixels for which it is not valid to get the pixel on top, bottom, left, right, ...
        no_top = _pixel_ids < _w
        no_bottom = _pixel_ids >= (_w * (_h - 1))
        no_left = _pixel_ids % _w == 0
        no_right = (_pixel_ids + 1) % _w == 0
        no_top_left = no_top + no_left
        no_top_right = no_top + no_right
        no_bottom_left = no_bottom + no_left
        no_bottom_right = no_bottom + no_right

        # 8-by-n matrix, where 'n' is the number of pixels identifiers
        neighbors = torch.stack([_pixel_ids - _w - 1,
                                 _pixel_ids - _w,
                                 _pixel_ids - _w + 1,
                                 _pixel_ids - 1,
                                 _pixel_ids + 1,
                                 _pixel_ids + _w - 1,
                                 _pixel_ids + _w,
                                 _pixel_ids + _w + 1])

        # fixing borders (replacing invalid neighbors with the identifier of the original pixels)
        neighbors[0, no_top_left] = _pixel_ids[no_top_left]
        neighbors[1, no_top] = _pixel_ids[no_top]
        neighbors[2, no_top_right] = _pixel_ids[no_top_right]
        neighbors[3, no_left] = _pixel_ids[no_left]
        neighbors[4, no_right] = _pixel_ids[no_right]
        neighbors[5, no_bottom_left] = _pixel_ids[no_bottom_left]
        neighbors[6, no_bottom] = _pixel_ids[no_bottom]
        neighbors[7, no_bottom_right] = _pixel_ids[no_bottom_right]

        return neighbors

    def __update_internal_supervision_counts(self, frame_was_already_considered=False):
        if self.__sup_added:
            if not frame_was_already_considered:
                self.create_supervision_count_checkpoint()
            else:
                self.restore_supervision_count_checkpoint()

            found_targets, targets_count = torch.unique(self.sup_buffer.get_last_frame_targets(), return_counts=True)
            for k in range(0, torch.numel(found_targets)):
                target = found_targets[k].item()
                assert self.get_class_name(target) is not None, \
                    ("ERROR: Target " + str(target) + " is not associated to any class names!")
                self.increment_supervision_count(target, num_sup=targets_count[k].item())

    def __update_avg_unsupervised_probs_over_time(self, unsupervised_probs):
        zeta = self.options['mi_history_weight']

        # moving average, weighed by 'zeta'
        if self.__avg_unsupervised_probs is not None and 0. <= zeta < 1.:
            avg_unsupervised_probs = zeta * torch.mean(unsupervised_probs, dim=0) \
                                     + (1.0 - zeta) * self.__avg_unsupervised_probs
        else:
            avg_unsupervised_probs = torch.mean(unsupervised_probs, dim=0)

        return avg_unsupervised_probs
