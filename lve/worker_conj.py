import os
import random
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform, randrange

import torchvision.utils
from torch.optim.lr_scheduler import StepLR

import lve
import torch
import cv2
import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import copy
from collections import OrderedDict

from lve.utils import backward_warp
from confeat.utils import sample_jitter_params, jitter_frames
import collections

class StreamBuffer():
    def __init__(self, options, ins, device):
        self.size = options['size']
        self.ins = ins
        self.buffer_idxs = [None] * self.size
        self.buffer_frame = [None] * self.size
        self.buffer_old_frame = [None] * self.size
        self.buffer_motion = [None] * self.size
        self.device = device

    def push(self, frame, old_frame, motion):
        if self.ins.current_repetition == 1:
            i = self.ins.get_last_frame_number() - 1
            if i < self.size:
                self.buffer_frame[i] = frame.cpu()
                self.buffer_old_frame[i] = old_frame.cpu()
                self.buffer_motion[i] = motion.cpu()
                self.buffer_idxs[i] = i
            else:
                j = random.randrange(i)
                if j < self.size:
                    self.buffer_frame[j] = frame.cpu()
                    self.buffer_old_frame[j] = old_frame.cpu()
                    self.buffer_motion[j] = motion.cpu()
                    self.buffer_idxs[j] = i

    def get(self, n):
        if self.ins.get_last_frame_number() < n:
            return None
        else:
            sampled_idxs = random.sample(range(min(self.size, self.ins.get_last_frame_number())), k=n)
            return {'frame': [self.buffer_frame[i].to(self.device) for i in sampled_idxs],
                    'old_frame': [self.buffer_old_frame[i].to(self.device) for i in sampled_idxs],
                    'motion': [self.buffer_motion[i].to(self.device) for i in sampled_idxs],
                    'sampled_idxs': sampled_idxs}

class UpdatePolicy():
    def __init__(self, options):
        self.length = options['length']
        self.skip = options['skip']

    def is_active(self, i):
        t = i % (self.length + self.skip)
        return t < self.length


class WorkerConj(lve.Worker):

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
        self.update_stage = 0

        # registering supported commands (commands that are triggered by the external visualizer/interface)
        self.register_command("reset_foa", self.__handle_command_reset_foa)
        self.register_command("supervise", self.__handle_command_supervise)

        self.ins = ins
        # saving a shortcut to the neural network options
        self.net_options = self.options["net"]
        self.net_options["w"] = self.w
        self.net_options["h"] = self.h
        self.update_policy = UpdatePolicy(self.net_options['update_policy']) if self.net_options['update_policy'][
                                                                                    'length'] != 0 else None

        # defining processors
        self.optical_flow = lve.OpticalFlowCV(backward=options["backward_optical_flow"])
        self.geymol = lve.GEymol(self.options["foa"], self.device) if self.options["foa"] is not None else None
        self.sup_buffer = lve.SuperWW(device=self.device)
        self.net = lve.NetConj(self.net_options, self.device, self.sup_buffer).to(self.device)

        """
         # neural network optimizer
        self.__lr = self.net_options["step_size"]
        if self.__lr < 0.:  # hack
            self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
        else:
            self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)"""

        # neural network optimizer

        self.__lr_features = self.net_options["step_size_features"]
        self.__lr_displacements = self.net_options["step_size_displacements"]

        self.displacements_params = [[p[1] for p in self.net.named_parameters() if str(j)+".displacements" in p[0]] for j in range(self.net_options["n_blocks"])]
        self.features_params = [[p[1] for p in self.net.named_parameters() if str(j)+".features" in p[0] and '_teacher.' not in p[0]] for j in range(self.net_options["n_blocks"])]

        if len(sum(self.features_params,[])) == 0:
            print("There are no params in any Features Block!")
            self.net_optimizer_features = None
        else:
            if self.__lr_features < 0.:  # hack
                self.net_optimizer_features = [torch.optim.Adam(self.features_params[j], lr=-self.__lr_features) for j in range(self.net_options["n_blocks"])]
            else:
                self.net_optimizer_features = [torch.optim.SGD(self.features_params[j], lr=self.__lr_features)  for j in range(self.net_options["n_blocks"])]
            if self.net_options["step_size_decay"] is not None:
                self.scheduler_features = [StepLR(self.net_optimizer_features[j], step_size=self.ins.effective_video_frames, gamma=self.net_options["step_size_decay"]) for j in range(self.net_options["n_blocks"])]
            else:
                self.scheduler_features = None

        if len(self.displacements_params) == 0:
            print("There are no params in any Displacement Block!")
            self.net_optimizer_displacements = None
        else:
            if self.__lr_displacements < 0.:  # hack
                self.net_optimizer_displacements = [torch.optim.Adam(self.displacements_params[j],
                                                                    lr=-self.__lr_displacements) for j in range(self.net_options["n_blocks"])]
            else:
                self.net_optimizer_displacements = [torch.optim.SGD(self.displacements_params[j],
                                                                   lr=self.__lr_displacements) for j in range(self.net_options["n_blocks"])]
            if self.net_options["step_size_decay"] is not None:
                self.scheduler_displacements = [StepLR(self.net_optimizer_displacements[j], step_size=self.ins.effective_video_frames, gamma=self.net_options["step_size_decay"])  for j in range(self.net_options["n_blocks"])]
            else:
                self.scheduler_displacements = None

        self.__freeze = self.net_options["freeze"]
        if self.__freeze:
            self.net.requires_grad_(False)

        if self.net_options['augmented_count'] > 0:
            assert self.net_options['microsaccades'] == 0
            assert self.net_options['jitters'] == 0
            assert self.net_options['flips'] == 0

        # misc
        self.reset()

        # setting up initial supervision map (if any)
        self.augment_supervision_map(self.options["supervision_map"], self.net_options["supervised_categories"])

        # flag to set eval forgetting
        self.eval_forgetting_flag = False

        self.loss_window = collections.deque(maxlen=self.ins.effective_video_frames)
        self.similarity_loss_window = collections.deque(maxlen=self.ins.effective_video_frames)

    def add_output_batched(self, outputs, keepdim=False):
        for k, v in outputs.items():
            # print(k, type(v))
            # if v is not None and v != []: # BUG when last dimension in numpy array is 1
            if type(v) == list and v != []:
                pass
            else:
                if v is not None:
                    if torch.is_tensor(v):
                        v = v.detach().cpu().numpy()
                    if type(v) == list:
                        for batch_i in range(len(v)):
                            self.add_output(k, v[batch_i], batch_i)
                    else:
                        for batch_i in range(v.shape[0]):
                            if keepdim:
                                self.add_output(k, v[None, batch_i], batch_i)
                            else:
                                self.add_output(k, v[batch_i], batch_i)

    def set_previous_frames_data(self, frame, of=None, supervisions=None, foa=None):
        if frame is not None and frame[0] is not None and frame[0][0] is not None:
            self.__fully_online = False
            b = len(frame)
            self.__old_frames = [None] * b
            for z in range(0, b):
                assert len(frame[z]) == 1, 'Right now we only process 1 frame from the past!'
                self.__old_frames[z] = frame[z][0]  # previous data is composed only by 1 frame data (mini-batch)

    def update_stats(self, d):
        self.__stats.update(d)

    def normalize_frames(self, frame):
        return (frame - 0.5) / 0.25


    def process_frame(self, frame, of=None, supervisions=None, foa=None, unfiltered_supervisions=None):
        # print("Processing frame {:} [{:}] (rep. {:}) with freeze={:},heavy_data={:} and sup={:}".format(
        #     self.ins.get_last_frame_number(),self.ins.get_last_frame_number_absolute(), self.ins.current_repetition, self.options['net']['freeze'], self.heavy_output_data_needed,
        #     self.get_supervision_count()['ewer']
        # ))
        # data returned by the call below and their types are:
        # frame: input frame (torch tensor, 1 x c x h x w, better keep the dummy batch dimension here),
        # motion: optical flow (torch tensor, 1 x 2 x h x w, better keep the dummy batch dimension here),
        # foa_row_col: focus of attention - integer coordinates (torch tensor, 2 elements)
        # saccade_bool: flag that tell if the movement of the focus of attention is a saccade (bool)
        # foa: focus of attention - real x, y, velocity_x, velocity_y (list of 4 elements)
        # sups: a pair of torch tensors (targets, indices), or (None, None)
        # if self.ins.get_last_frame_number() == 5:
        #     print('ciao')
        #     print(time.time())
        #     self.__initial_time = time.time()

        # print a random weight from the feature block
        # print(f"Current frame: {self.ins.get_last_frame_number()}")
        # print([a for a in self.net.named_parameters()][2][1].detach().cpu().numpy()[0][0])  # for convblock
        # print([a for a in self.net.named_parameters()][2][1].detach().cpu().numpy())  # for resunetnnblock_bias
        # if self.ins.get_last_frame_number() == 200:
        #     sdfsdfg

        frame, motion, foa_row_col, saccade_bool, foa, sup_targets, sup_indices, self.__old_frames = \
            self.__compute_missing_data_and_convert(frame, of, foa, supervisions, self.__old_frames)


        # WAIT_FRAMES = 2000
        # if self.ins.get_last_frame_number() == WAIT_FRAMES:
        #     print((time.time() - self.__initial_time) / WAIT_FRAMES)
        #     dfnjsdkfnjkds

        sigma = 0
        frame_mean = torch.zeros(1).to(frame.device)

        if self.__old_frames is None:
            self.__old_frames = frame
            self.__old_frames_mean = frame_mean

        """
                        # temp hack for supervisions plot

                        
        if supervisions is not None:
            for hhh in range(0, len(frame_BKP)):
                if supervisions[hhh] is not None and supervisions[hhh][0] is not None:
                    path_sup = "supervision_folder_empty_space"
                    os.makedirs(path_sup, exist_ok=True)
                    plt.imshow(frame_BKP[hhh][..., ::-1])
                    row = supervisions[hhh][1][0] // self.w
                    col = supervisions[hhh][1][0] % self.h
                    plt.plot([col], [row], marker="x")
                    plt.title(f"Class: {supervisions[hhh][0][0]}")
                    plt.savefig(os.path.join(path_sup, f"class_{str(supervisions[hhh][0][0])}_idx_{self.counter_sup}"))
                    self.counter_sup = self.counter_sup + 1
                    plt.close()
        """

        """
        # temp hack for color wheel plot
        samples = 256
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))

        stacked_grid = np.stack((xx, yy), axis=2)
        of = [stacked_grid]
        """

        # we override saccades with a flag that is only about movement
        b = frame.shape[0]
        saccade_bool = [torch.norm(motion[x, :, foa_row_col[x, 0], foa_row_col[x, 1]]) == 0 for x in range(0, b)]

        if self.net_options['augmented_count'] > 0:
            r = [random.random() for i in range(3)]
            s = sum(r)
            r = [round(i / s *  self.net_options['augmented_count']) for i in r]
            if r[0] > 3:
                r[0] = 3
                r[1] = round(random.random() * (self.net_options['augmented_count'] - 3))
                r[2] = self.net_options['augmented_count'] - r[0] - r[1]
            self.net_options["flips"], self.net_options["microsaccades"], self.net_options["jitters"] = r

        if self.net_options["normalize"]:
            # normalization (customizable)
            frame_not_normalized = frame if self.net_options["jitters"] > 0 else None
            old_frame_not_normalized = self.__old_frames if self.net_options["jitters"] > 0 else None
            frame = self.normalize_frames(frame)
            if not self.__fully_online:
                self.__old_frames = self.normalize_frames(self.__old_frames)
        else:
            frame_not_normalized = frame
            old_frame_not_normalized = self.__old_frames

            # sampling a/some buffered frame(s) (associated to previous supervisions - if any)
        buff_frames = self.sup_buffer.sample_frames(self.options['piggyback_frames'],
                                                    self.options['sup_persistence'])
        buff_frames = buff_frames.to(self.device) if buff_frames is not None else None

        old_frames_batch = self.__old_frames
        frame_batch = frame

        # list to be concatenated at the end of transforms - initialize them with the actual frame and old_frame
        old_frames_batch_list = [old_frames_batch]
        frame_batch_list = [frame_batch]
        transformed_motion = [motion]

        # add augmented frames if flips are active
        if self.net_options["flips"] > 0:
            assert self.net_options["flips"] <= 3, "Maximum number of augmented data flips is 3."
            dims = [[2], [3], [2, 3]]
            random.shuffle(dims)

            # flipped_frame_batch = [frame_batch, torch.flip(frame_batch, dims=dims[0])]
            frame_batch_list.append(torch.flip(frame_batch, dims=dims[0]))
            # flipped_old_frames_batch = [old_frames_batch, torch.flip(old_frames_batch, dims=dims[0])]
            old_frames_batch_list.append(torch.flip(old_frames_batch, dims=dims[0]))
            if motion is not None and isinstance(motion, torch.Tensor):
                motion_f = torch.flip(motion, dims=dims[0])
                if len(dims[0]) == 1 and dims[0][0] == 2:
                    motion_f[:, 1, :, :] = -motion_f[:, 1, :, :]
                elif len(dims[0]) == 1 and dims[0][0] == 3:
                    motion_f[:, 0, :, :] = -motion_f[:, 0, :, :]
                else:
                    motion_f = -motion_f
                transformed_motion.append(motion_f)

            if self.net_options["flips"] > 1:
                frame_batch_list.append(torch.flip(frame_batch, dims=dims[1]))
                old_frames_batch_list.append(torch.flip(old_frames_batch, dims=dims[1]))
                if motion is not None and isinstance(motion, torch.Tensor):
                    motion_f = torch.flip(motion, dims=dims[1])
                    if len(dims[1]) == 1 and dims[1][0] == 2:
                        motion_f[:, 1, :, :] = -motion_f[:, 1, :, :]
                    elif len(dims[1]) == 1 and dims[1][0] == 3:
                        motion_f[:, 0, :, :] = -motion_f[:, 0, :, :]
                    else:
                        motion_f = -motion_f
                    transformed_motion.append(motion_f)

                if self.net_options["flips"] > 2:
                    frame_batch_list.append(torch.flip(frame_batch, dims=dims[2]))
                    old_frames_batch_list.append(torch.flip(old_frames_batch, dims=dims[2]))
                    if motion is not None and isinstance(motion, torch.Tensor):
                        motion_f = torch.flip(motion, dims=dims[2])
                        if len(dims[2]) == 1 and dims[2][0] == 2:
                            motion_f[:, 1, :, :] = -motion_f[:, 1, :, :]
                        elif len(dims[2]) == 1 and dims[2][0] == 3:
                            motion_f[:, 0, :, :] = -motion_f[:, 0, :, :]
                        else:
                            motion_f = -motion_f
                        transformed_motion.append(motion_f)



        # add augmented frames if microsaccades are active
        if self.net_options["microsaccades"] > 0:
            micro_dict = {"removed_percentage": self.net_options["removed_percentage"]} #  CHECK 0.1!
            old_frames_b, frame_b = self.apply_transformation_frame_pair(frame=frame_batch, old_frame=old_frames_batch,
                                                                         fn=self.__microsaccade_crop,
                                                 nums_transforms=self.net_options["microsaccades"], params=micro_dict)

            old_frames_batch_list.append(old_frames_b)
            frame_batch_list.append(frame_b)


            # also frame must be a batch of repeated frames, but must be handled the fact that sup_buffer uses it
            # so keep intact the "frame" variable! (in the code above)

        if self.net_options["jitters"] > 0:
            brightness = (0.7, 1.5)
            contrast = (0.8, 1.2)
            saturation = (0.4, 1.6)
            hue = (-0.3, 0.3)
            gaussian = (0.1, 2.0)
            jitter_dict = {"brightness": brightness,
                           "contrast": contrast,
                           "saturation": saturation,
                           "hue": hue,
                           "gaussian": gaussian}
            old_frames_b_jit, frame_b_jit = self.apply_transformation_frame_pair(frame=frame_not_normalized,
                                                                         old_frame=old_frame_not_normalized,
                                                                         fn=self.__jitter,
                                                                         nums_transforms=self.net_options[
                                                                             "jitters"], params=jitter_dict)
            old_frames_batch_list.append(old_frames_b_jit)
            frame_batch_list.append(frame_b_jit)
            transformed_motion.extend([motion for _ in range(self.net_options["jitters"])])

            # pass  # obiettivo aggiungere cose a "frame_batch" e a "old_frames_batch",
            # jitterando "frame_not_normalized" e da "old_frame_not_normalized"
            # con la logica di microsaccades (quasi identico)
            # poi eventualmente vanno rinormalizzati

        # now concatenate all the transformed samples!
        frame_batch = torch.cat(frame_batch_list, dim=0)
        old_frames_batch = torch.cat(old_frames_batch_list, dim=0)

        if motion is not None and isinstance(motion, torch.Tensor):
            motion = torch.cat(transformed_motion, dim=0)

        # inference (full inference on the current frame, encoding only on the buffered frames - if any)
        #   the current frame and the buffered frames constitute a batch of frames (index 0: current frame)
        # notice: features_current, displacements are lists

        # debug visualize:
        # lve.utils.plot_grid(frame_batch.detach().cpu().permute(0,2,3,1).flip(3).numpy())
        # plt.imshow(frame_batch[0].permute(1,2,0).flip(-1).detach().cpu().numpy())

        augmented_count = self.net_options["microsaccades"] + self.net_options["flips"] + self.net_options["jitters"]
        features_current, features_old, lower_features_current, lower_features_old, format_required_raw_features, format_required_features, format_required_piggyback, \
        displacements, supervised_probs_masked_list, prediction_mask_list, prediction_idx_list, unmasked_prediction_idx, format_required_old_features, \
        logits_current, logits_old = \
            self.net(frame_batch, old_frames_batch, buff_frames, motion,
                     augmented_count=augmented_count)

        if self.net_options["jitters"] > 0:
            for i in range(len(displacements['fwd'])):
                    b_real = int(frame_batch.shape[0] // int(augmented_count + 1))
                    displacements['fwd'][i][-self.net_options["jitters"]*b_real:] = displacements['fwd'][i][:b_real].repeat(self.net_options["jitters"], 1, 1, 1)

        self.__frame_embeddings = format_required_raw_features
        piggyback_frame_embeddings = format_required_piggyback

        # updating embeddings of the supervised pixel(s) of the buffered frames (if any)
        self.sup_buffer.update_embeddings_of_sampled_frames(piggyback_frame_embeddings.detach()
                                                            if piggyback_frame_embeddings.shape[0] > 0 else None)

        for x in range(0, b):
            # storing supervisions associated to the current frame, provided by the input stream (if any)
            self.__sup_added = self.sup_buffer.add(frame[None, x, :, :, :], sup_targets[x], sup_indices[x],
                                                   self.__frame_embeddings[None, x, :, :, :])
            # updating the worker-level information about supervision (worker has its own counts, exposed to visualizer)
            self.__update_internal_supervision_counts()

            sup_targets[x] = self.sup_buffer.get_last_frame_targets().cpu().numpy().astype(
                np.uint32) if self.__sup_added else None
            sup_indices[x] = self.sup_buffer.get_last_frame_indices().cpu().numpy().astype(
                np.uint32) if self.__sup_added else None

        # detach motion to be given as input to the decoder?
        detached_displacements = {'fwd': [displ.clone().detach() for displ in displacements['fwd']],
                                  'bwd': [displ.clone().detach() for displ in displacements['bwd']]}

        predicted_frames = []

        l = self.net.whole_model.compute_loss(features_current, features_old, lower_features_current,
                                              lower_features_old,
                                              displacements, frame_batch,
                                              old_frame=old_frames_batch,
                                              predicted_frame=predicted_frames)

        self.loss_window.appendleft(l["total"].item())
        self.similarity_loss_window.appendleft(l["similarity_loss"].item())

        self.__unsup_loss = l["total"]

        # sampling supervised data (if any)
        embeddings, labels = self.sup_buffer.sample_embeddings_batch(self.options['sup_batch'],
                                                                     self.options['sup_persistence'])
        embeddings = embeddings.to(self.device) if embeddings is not None else None
        labels = labels.to(self.device) if labels is not None else None

        # loss (supervised)
        self.__sup_loss, sup_details = self.net.compute_supervised_loss(embeddings, labels)

        supervised_probs_masked_list = [el.detach().permute(0, 2, 1). \
                                            view(b, el.shape[2], self.h, self.w).cpu().numpy() for el in
                                        supervised_probs_masked_list]

        prediction_idx_list_detached = [el.detach().cpu().numpy() for el in prediction_idx_list]

        self.__stats.update({
            "loss": self.__unsup_loss.item() + self.__sup_loss.item(),
            "consistency_lower": l["consistency_lower"].item(),
            "consistency_upper": l["consistency_upper"].item(),
            "motion_smoothness": l["motion_smoothness"].item(),
            "spatial_coherence": l["spatial_coherence"].item(),
            "similarity_loss": l["similarity_loss"].item(),
            "foax": foa[0, 0].item(), "foay": foa[0, 1].item(),  # dummy data from the 1st mini-batch elem
            "foavx": foa[0, 2].item(), "foavy": foa[0, 3].item(),  # dummy data from the 1st mini-batch elem
            "saccade": int(saccade_bool[0].item())  # dummy data from the 1st mini-batch elem
            # "current_threshold": self.net_options["net"]["dist_threshold"]
        }
        )
        self.__stats.update(
            {f"similarity_b{i}": l[f"similarity_b{i}"].item() for i in range(self.net_options["n_blocks"])})
        self.__stats.update(
            {f"consistency_lower_b{i}": l[f"consistency_lower_b{i}"].item() for i in
             range(self.net_options["n_blocks"])})
        self.__stats.update(
            {f"consistency_skip_b{i}": l[f"consistency_skip_b{i}"].item() for i in
             range(self.net_options["n_blocks"])})
        self.__stats.update(
            {f"consistency_upper_b{i}": l[f"consistency_upper_b{i}"].item() for i in
             range(self.net_options["n_blocks"])})

        if self.heavy_output_data_needed:
            features_numpy = self.__frame_embeddings.detach().cpu().numpy()
            old_features_numpy = format_required_old_features.detach().cpu().numpy()

            warped_features = backward_warp(frame=self.__frame_embeddings, displacement=displacements['fwd'][-1][0:b])
            warped_features = warped_features.detach().cpu().numpy()
            if self.net_options["normalize"]:
                old_frames_to_visualize = self.__old_frames.detach() * 0.25 + 0.5
            else:
                old_frames_to_visualize = self.__old_frames.detach()
            old_frames_to_visualize = lve.utils.torch_float_01_to_np_uint8(old_frames_to_visualize +
                                                                           (self.__old_frames_mean.detach()
                                                                            if self.__old_frames_mean is not None else 0.))
        else:
            features_numpy = old_features_numpy = warped_features = old_frames_to_visualize = None

        if self.heavy_output_data_needed:
            self.add_output_batched({
                "features": features_numpy,  # bin
                "old_features": old_features_numpy,  # bin
                "warped_features": warped_features,  # bin
            }, keepdim=True)
            self.add_output_batched({
                "old_frames": old_frames_to_visualize,  # bin
            }, keepdim=False)

            # network predicted motions
            for i in range(0, len(displacements['fwd'])):
                self.add_output_batched({
                    "dflow_x_2d." + str(i): l['d_flow_x_2d'][i],
                    "dfeat_x_2d." + str(i): l['d_feat_x_2d'][i],
                    "dflow_y_2d." + str(i): l['d_flow_y_2d'][i],
                    "dfeat_y_2d." + str(i): l['d_feat_y_2d'][i],
                    "net_motion." + str(i): displacements['fwd'][i],
                    # "m_2d." + str(i): l['m_2d'][i],
                }, keepdim=True)

                if 'simdissim_points' in l:
                    self.add_output_batched({"simdissim_points." + str(i): l['simdissim_points'][i]})

                def add_output_batched_from_l(l, key):
                    if key in l and l[key] is not None and l[key][i] is not None and l[key][i] != []:
                        self.add_output_batched({key + "." + str(i): l[key]}, keepdim=True)

                for k in ['m_2d', 'closeness_2d', 'fsmoothness_2d', 'super_features_2d', 'out_features_2d',
                          'cvx_2d', 'cvy_2d', 'dvx_2d', 'dvy_2d', 'cfx_2d', 'cfy_2d', 'dfx_2d', 'dfy_2d',
                          'consistency_upper_2d',
                          ]:
                    add_output_batched_from_l(l, k)

            for x in range(0, b):
                self.add_outputs({"motion": of[x]})
                self.__stats.update({"foax": foa[x, 0].item(), "foay": foa[x, 1].item(),
                                     "foavx": foa[x, 2].item(), "foavy": foa[x, 3].item(),
                                     "saccade": int(saccade_bool[x])})

                for i in range(0, len(predicted_frames)):
                    pred_frames_uint8 = lve.utils.torch_float_01_to_np_uint8(
                        torch.clamp(predicted_frames[i][x].detach(), min=0, max=1).cpu())
                    self.add_output("predicted_frames." + str(i), pred_frames_uint8, batch_index=x)

                    if sum(self.options['net']['lambda_p_masked']) > 0.:
                        target_frame = l[f"masked_target_b{i}"][x]
                        masked_old_frames_to_visualize = lve.utils.torch_float_01_to_np_uint8(target_frame)
                        self.add_output("masked_target_b" + str(i), masked_old_frames_to_visualize, batch_index=x)

                # NB The  Following code solves the problem of the artifatcs in wandb, but the frame in lve is total black

                # tensor_max = predicted_frames[i][0].detach().cpu().max()
                # tensor_min = predicted_frames[i][0].detach().cpu().min()
                # tensor_norm = (predicted_frames[i][0].detach().cpu() - tensor_min) / (tensor_max - tensor_min)
                # self.add_output("predicted_frames." + str(i),
                #                 tensor_norm.numpy().transpose(1, 2, 0))  # network predicted frames
            self.add_outputs({"wheel": self.__wheel})
            for x in range(0, b):
                supervised_probs_masked_list_x = [el[None, x] for el in supervised_probs_masked_list]
                prediction_idx_list_detached_x = [el[x] for el in prediction_idx_list_detached]

                self.add_outputs({"unmasked-prediction_idx": unmasked_prediction_idx[x].detach().cpu().numpy(),
                                  "sup-probs": supervised_probs_masked_list_x[0],
                                  # bin (first threshold, that's why [0])
                                  "sup-probs-list": supervised_probs_masked_list_x,  # bin
                                  "prediction_idx": prediction_idx_list_detached_x[0],
                                  # (first threshold, that's why [0])
                                  "prediction_idx-list": prediction_idx_list_detached_x,
                                  "sup.targets": sup_targets[x],  # bin
                                  "sup.indices": sup_indices[x],  # bin
                                  },  # CSV log
                                 batch_index=x)

        self.add_outputs({
            "window_loss": np.sum(self.loss_window) / len(self.loss_window),
            "window_similarity_loss": np.sum(self.similarity_loss_window) / len(self.similarity_loss_window),
        })

        self.add_outputs({
            "sup.map": self.get_supervision_map() if self.__sup_new_category_added else None,  # JSON
            "stats.worker": copy.deepcopy(self.__stats),  # dictionary
            "logs.worker": list(self.__stats.values())})




        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        self.__sup_new_category_added = False
        self.__frame = frame

        if self.__fully_online:
            self.__old_frames = frame
            self.__old_frames_mean = frame_mean
        # self.__foa = (foa[0], foa[1])

    def apply_transformation_frame_pair(self, frame, old_frame, fn, nums_transforms, params):

        add_extra = nums_transforms % 2 == 1
        extra_is_old = np.random.randint(2) == 0
        num_old = (nums_transforms // 2) + (add_extra and extra_is_old)
        num_cur = (nums_transforms // 2) + (add_extra and not extra_is_old)

        transformed_batch_old = None
        if num_old > 0:
            transformed_batch_old = fn(old_frame, num_old, params)

        transformed_batch_cur = None
        if num_cur > 0:
            transformed_batch_cur = fn(frame, num_cur, params)

        if transformed_batch_old is not None and transformed_batch_cur is not None:
            # old_frames_batch = torch.cat((old_frame,
            #                               transformed_batch_old,
            #                               old_frame.repeat(num_cur, 1, 1, 1)), dim=0)
            # frame_batch = torch.cat((frame.repeat(1 + num_old, 1, 1, 1),
            #                          transformed_batch_cur), dim=0)
            old_frames_batch = torch.cat((transformed_batch_old,
                                          old_frame.repeat(num_cur, 1, 1, 1)), dim=0)
            frame_batch = torch.cat((frame.repeat(num_old, 1, 1, 1),
                                     transformed_batch_cur), dim=0)

        elif transformed_batch_old is not None:
            # old_frames_batch = torch.cat((old_frame,
            #                               transformed_batch_old), dim=0)
            # frame_batch = frame.repeat(1 + num_old, 1, 1, 1)
            old_frames_batch = transformed_batch_old
            frame_batch = frame.repeat(num_old, 1, 1, 1)
        elif transformed_batch_cur is not None:
            # old_frames_batch = old_frame.repeat(1 + num_cur, 1, 1, 1)
            # frame_batch = torch.cat((frame, transformed_batch_cur), dim=0)
            old_frames_batch = old_frame.repeat(num_cur, 1, 1, 1)
            frame_batch = transformed_batch_cur
        else:
            raise RuntimeError("Unexpected!")

        return old_frames_batch, frame_batch # we return only the additional transformed

    def export_internal_data(self):
        return {'__sup_new_category_added': self.__sup_new_category_added,
                '__frame': self.__frame,
                '__old_frames': self.__old_frames,
                '__old_frames_mean': self.__old_frames_mean
                }

    def set_internal_data(self, data):
        for k, v in data.items():
            setattr(self, k, v)

    def set_eval_forgetting(self):
        self.eval_forgetting_flag = True

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
        loss = (self.__unsup_loss + self.__sup_loss)

        if loss.requires_grad:
            update_flag = self.update_policy is None or self.update_policy.is_active(self.ins.get_last_frame_number())
            # pythonprint('f', self.ins.get_last_frame_number(), 'update_flag', update_flag)
            if update_flag:
                # computing gradients
                loss.backward()

                if self.net_options['gradient_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.net_options['gradient_clip'])

                # update step
                def perform_step(optimizers, schedulers, block=None):
                    if block is not None:
                        optimizers = [optimizers[block]]
                        if schedulers is not None: schedulers = [schedulers[block]]
                    if optimizers is not None:
                        [optimizer.step() for optimizer in optimizers]
                        if schedulers is not None: [scheduler.step() for scheduler in schedulers]

                if self.net_options["block_scheduling"] is None or self.net_options["block_scheduling"]['step'] == 0:
                    perform_step(self.net_optimizer_features, self.scheduler_features)
                    perform_step(self.net_optimizer_displacements, self.scheduler_displacements)
                else:
                    update_stage = (self.ins.current_repetition-1) // self.net_options["block_scheduling"]['step']
                    # update_stage 0 -> moto-0
                    # update_stage 1 -> feature-0
                    # update_stage 2 -> moto-1
                    # update_stage 3 -> feature-1
                    log_update = False
                    if update_stage > self.update_stage and update_stage < 2 * self.net_options['n_blocks']:
                        print('** Entering stage', update_stage)
                        log_update = True
                    for j in range(self.net_options['n_blocks']):
                        if self.net_options["block_scheduling"]['mode'] == 'all':
                            if update_stage // 2 >= j:
                                if log_update: print('** updating displacements-'+str(j))
                                perform_step(self.net_optimizer_displacements, self.scheduler_displacements, block=j)
                            if update_stage > 0 and (update_stage-1) // 2 >= j:
                                if log_update: print('** updating features-' + str(j))
                                perform_step(self.net_optimizer_features, self.scheduler_features, block=j)
                        elif self.net_options["block_scheduling"]['mode'] == 'features':
                            perform_step(self.net_optimizer_displacements, self.scheduler_displacements)
                            if self.ins.current_repetition >= self.net_options["block_scheduling"]['step']: perform_step(self.net_optimizer_features, self.scheduler_features)
                    self.update_stage = update_stage
            self.net.zero_grad()

        if self.net_options['teacher']:
            self.net.whole_model.update_teacher_features_blocks()

        # detaching last frame supervisions (if any)
        if self.__sup_added:
            self.sup_buffer.detach_last_frame_supervisions()

        # check if learning rate was changed (hot)
        # if self.__lr != self.net_options['step_size']:
        #     self.__lr = self.net_options['step_size']
        #     if self.__lr < 0.:
        #         self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
        #     else:
        #         self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)

        # check if learning rate was changed (hot)
        if self.__lr_features != self.net_options['step_size_features'] or self.__lr_displacements != self.net_options[
            'step_size_displacements']:
            self.__lr_features = self.net_options['step_size_features']
            self.__lr_displacements = self.net_options['step_size_displacements']

            if self.__lr_features < 0.:  # hack
                self.net_optimizer_features = [torch.optim.Adam(self.features_params[j], lr=-self.__lr_features) for j in self.net_options['n_blocks']]
            else:
                self.net_optimizer_features = [torch.optim.SGD(self.features_params[j], lr=self.__lr_features) for j in self.net_options['n_blocks']]

            if self.__lr_displacements < 0.:  # hack
                self.net_optimizer_displacements = [torch.optim.Adam(self.displacements_params[j],
                                                                    lr=-self.__lr_displacements) for j in self.net_options['n_blocks']]
            else:
                self.net_optimizer_displacements = [torch.optim.SGD(self.displacements_params[j],
                                                                   lr=self.__lr_displacements) for j in self.net_options['n_blocks']]

    def load(self, model_folder):
        print('Loading model in worker...')
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth", map_location=self.device))

        if os.path.exists(worker_model_folder + "worker.pth"):
            # loading worker-status related tensors
            worker_status = torch.load(worker_model_folder + "worker.pth", map_location=self.device)
            self.__activations_foa_prev = worker_status["what_for_prev"]
            self.__avg_unsupervised_probs = worker_status["avg_unsupervised_probs"]
            self.sup_buffer = worker_status["supervision_buffer"]

        if os.path.exists(worker_model_folder + "worker.json"):
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

    def load_weights(self, model_folder):
        print('Loading model in worker...')
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth", map_location=self.device))

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
            "wheel": {'data_type': lve.OutputType.MOTION, 'per_frame': True},
            "features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "warped_features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "old_features": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "old_frames": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "blob": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sampled-points": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "unsup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "unsup-probs_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "unmasked-prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "prediction_idx-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "pred-mask": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sup.indices": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.targets": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.map": {'data_type': lve.OutputType.JSON, 'per_frame': False},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys()),  # first line of CSV
            "window_loss": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "window_similarity_loss": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
        }

        for i in range(0, self.options['net']['n_blocks']):
            output_types.update({"occl_2d_lower." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"occl_2d_upper." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"dflow_x_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"dfeat_x_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"dflow_y_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"dfeat_y_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"fsmoothness_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"closeness_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"simdissim_points." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"net_motion." + str(i): {'data_type': lve.OutputType.MOTION, 'per_frame': True}})
            output_types.update({"m_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update(
                {"super_features_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"out_features_2d." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"free_features." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
            output_types.update({"free_motion." + str(i): {'data_type': lve.OutputType.MOTION, 'per_frame': True}})
            for x in ['cvx_2d', 'cvy_2d', 'dvx_2d', 'dvy_2d', 'cfx_2d', 'cfy_2d', 'dfx_2d', 'dfy_2d',
                      'consistency_upper_2d']:
                output_types.update({x + "." + str(i): {'data_type': lve.OutputType.BINARY, 'per_frame': True}})

        return output_types

    def print_info(self):
        s = "   worker {"
        i = 0
        for k, v in self.__stats.items():
            if not isinstance(v, str):
                s += (k + (": {0:.8e}".format(v) if abs(v) >= 1000 else ": {0:.8f}".format(v)))
            else:
                s += (k + ": " + v)
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

    def reset(self):
        super().reset()
        self.__ids = torch.arange(self.w * self.h, device=self.device)
        self.__activations_foa_prev = None
        self.__avg_unsupervised_probs = None
        self.__sup_added = False
        self.__sup_new_category_added = False
        self.__frame = None
        self.__old_frames = None
        self.__frame_embeddings = None
        self.__motion_needed = (self.geymol is not None and self.geymol.parameters["alpha_of"] > 0.)
        self.__unsup_loss = torch.tensor(-1.).to(self.device)
        self.__sup_loss = torch.tensor(-1.).to(self.device)
        self.__stats = OrderedDict(
            [('loss', -1.), ('consistency_lower', -1.), ('consistency_upper', -1.),('motion_smoothness', -1.),
             ('foax', -1.), ('foay', -1.),
             ('saccade', -1)] + [(f'consistency_lower_b{i}', -1) for i in
                                                                 range(self.net_options['n_blocks'])] + [
                (f'consistency_upper_b{i}', -1) for i in range(self.net_options['n_blocks'])] +
            [(f'consistency_skip_b{i}', -1) for i in range(self.net_options['n_blocks'])]
        )  # placeholders
        # self.__foa = None

        self.__fully_online = True
        self.__old_frames_mean = None

        self.counter_sup = 0
        along_h, along_w = torch.meshgrid(torch.arange(self.h), torch.arange(self.w), indexing='ij')
        along_h = 2. * (along_h / float(self.h - 1)) - 1.
        along_w = 2. * (along_w / float(self.w - 1)) - 1.
        along_hw = torch.stack((along_w, along_h), dim=0)  # motion is: (horizontal motion, vertical motion)
        self.__wheel = along_hw.unsqueeze(0)  # 1 x 2 x h x w
        self.__wheel = self.__wheel.numpy()

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

    def __jitter(self, img, jitters, params):
        jitters_batch = []
        for j in range(jitters):
            b, c, s, h, g = sample_jitter_params(brightness=params['brightness'],
                                              contrast=params['contrast'],
                                              hue=params['hue'],
                                              saturation=params['saturation'],
                                              gaussian=params['gaussian'])
            tr_frames = jitter_frames(img, b, c, s, h, g)
            jitters_batch.append(tr_frames)

        jitters_batch = torch.cat(jitters_batch, dim=0)
        if self.net_options["normalize"]:
            jitters_batch = self.normalize_frames(jitters_batch)
        return jitters_batch



    def __microsaccade_crop(self, img, microsaccades, params):

        removed_percentage = params["removed_percentage"]
        assert 0 <= removed_percentage < 1, "Invalid percentage!"

        microsaccade_batch = []
        for j in range(microsaccades):
            max_pixel_removed = int(removed_percentage * min(img.shape[2], img.shape[3]))
            a, b = torch.randint(0, max_pixel_removed, (2,), device=self.device)
            a, b = a - max_pixel_removed // 2, b - max_pixel_removed // 2  # (a columns, b rows)
            top, left, height, width = 0, 0, img.shape[2], img.shape[3]

            if b > 0:
                top = top + b
                height = height - b
            else:
                height = height + b  # b is negative
            if a > 0:
                left = left + a
                width = width - a
            else:
                width = width + a  # a is negative
            microsaccade_batch.append(
                transforms.functional.resized_crop(img, top=top, left=left, height=height, width=width,
                                                   size=img.shape[2:]))

        return torch.cat(microsaccade_batch, dim=0)

    def __compute_missing_data_and_convert(self, batch_frames_np_uint8, batch_motion_np_float32,
                                           batch_foa_np_float32, batch_sup_np,
                                           batch_prev_frames_np_uint8):

        batch_size = len(batch_frames_np_uint8)
        batch_frames_torch = [None] * batch_size
        batch_motion_torch = [None] * batch_size
        batch_foa_torch = [None] * batch_size
        batch_foa_row_col_torch = [None] * batch_size
        batch_sup_targets_torch = [None] * batch_size
        batch_sup_indices_torch = [None] * batch_size
        batch_saccade_bool = [None] * batch_size
        batch_prev_frames_torch = [None] * batch_size

        for i in range(0, batch_size):

            # convert to tensor
            frame_np_uint8 = batch_frames_np_uint8[i]
            frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8)
            batch_frames_torch[i] = frame

            if batch_prev_frames_np_uint8 is not None and not isinstance(batch_prev_frames_np_uint8, torch.Tensor):
                batch_prev_frames_torch[i] = lve.utils.np_uint8_to_torch_float_01(batch_prev_frames_np_uint8[i])

            # grayscale-instance of the input frame
            if not self.frame_is_gray_scale:
                frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
                frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8)
            else:
                frame_gray_np_uint8 = frame_np_uint8
                frame_gray = frame

            # optical flow
            if batch_motion_np_float32 is None or batch_motion_np_float32[0] is None:
                if self.__motion_needed:
                    motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
                else:
                    motion_np_float32 = np.zeros((self.h, self.w, 2), dtype=np.float32)
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32)  # 1 x 2 x h x w

                if batch_motion_np_float32 is not None:
                    batch_motion_np_float32[i] = motion_np_float32  # updating, it might be used out of this function
            else:
                motion_np_float32 = batch_motion_np_float32[i]  # h x w x 2
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32)  # 1 x 2 x h x w

            batch_motion_torch[i] = motion

            # focus of attention (a negative dissipation triggers a random trajectory, useful in some experiments)
            if batch_foa_np_float32 is None or batch_foa_np_float32[i] is None:
                if self.geymol is not None:
                    if self.options["foa"]["dissipation"] >= 0.0:
                        foa, saccade_bool = self.geymol.next_location(frame_gray.to(self.device),
                                                                      motion.to(self.device),
                                                                      frame_gray_uint8_cpu=frame_gray_np_uint8)
                    else:
                        foa = np.array([randrange(0, self.h - 1), randrange(0, self.w - 1), 0., 0.], dtype=np.float64)
                        saccade_bool = False
                else:
                    foa = np.array([0., 0., 0., 0.], dtype=np.float64)
                    saccade_bool = False
            else:
                foa = batch_foa_np_float32[i][0:4]
                saccade_bool = bool(batch_foa_np_float32[i][-1])

            batch_foa_torch[i] = torch.from_numpy(foa)
            batch_saccade_bool[i] = saccade_bool

            # getting the two integer coordinates of the focus of attention (discarding velocity)
            foa_row_col = torch.from_numpy(foa[0:2].astype(np.compat.long)).view(2).to(torch.long)
            batch_foa_row_col_torch[i] = foa_row_col

            # a pair of (targets, indices)
            if batch_sup_np is not None and batch_sup_np[i] is not None:
                sup_targets, sup_indices = torch.tensor(batch_sup_np[i][0], dtype=torch.int64), torch.tensor(
                    batch_sup_np[i][1])
            else:
                sup_targets, sup_indices = None, None

            batch_sup_targets_torch[i] = sup_targets
            batch_sup_indices_torch[i] = sup_indices

        # stacking into tensors (not all of them)
        frame = torch.cat(batch_frames_torch, dim=0).to(self.device)
        motion = torch.cat(batch_motion_torch, dim=0).to(self.device)
        foa_row_col = torch.stack(batch_foa_row_col_torch, dim=0)  # CPU
        foa = torch.stack(batch_foa_torch, dim=0)  # CPU

        if batch_prev_frames_np_uint8 is not None:
            if not isinstance(batch_prev_frames_np_uint8, torch.Tensor):
                prev_frame = torch.cat(batch_prev_frames_torch, dim=0).to(self.device)
            else:
                prev_frame = batch_prev_frames_np_uint8
        else:
            prev_frame = None

        return frame, motion, foa_row_col, batch_saccade_bool, foa, \
               batch_sup_targets_torch, batch_sup_indices_torch, prev_frame

    def __compute_foa_moving_blob(self, motion, foa_row_col, motion_threshold=0.1):
        # old_code
        if motion_threshold < 0:
            moving_bool = torch.sum(motion ** 2, dim=1).view(self.h, self.w) > 0
            if moving_bool[foa_row_col[0], foa_row_col[1]]:
                return moving_bool
            else:
                return torch.zeros((self.h, self.w), dtype=torch.bool, device=motion.device)

        """
        # batch tentative code
        b = motion.shape[0]
        if motion_threshold < 0:
            moving_bool = torch.sum(motion ** 2, dim=1).view(b, self.h, self.w) > 0    # mask true if pixel moving
            # now must detect if foa inside a moving pixel
            # https://stackoverflow.com/questions/65815668/how-to-select-indices-according-to-another-tensor-in-pytorch
            # is_foa_in = torch.gather(moving_bool, dim=1, index=foa_row_col)  # output deve essre [b, 1]
            moving_bool_flat = moving_bool.flatten(start_dim=1)  # flattened
            foa_frontier = foa_row_col[:, 0, None] * self.w + foa_row_col[:, 1, None]  # flattened coordinates of the FOA
            is_foa_in = torch.gather(moving_bool_flat, dim=1, index=foa_frontier)
            return moving_bool * is_foa_in.unsqueeze(1)  # CHECK! should put all tensor dim to zero
            # for batch in range(b):  # brutto!
            #     if not moving_bool[batch, foa_row_col[batch, 0], foa_row_col[batch, 1]]:
            #         moving_bool[batch] = torch.zeros((self.h, self.w), dtype=torch.bool, device=motion.device)
            #
            # if moving_bool[foa_row_col]:
            #     return moving_bool
            # else:
            #     return torch.zeros((b, self.h, self.w), dtype=torch.bool, device=motion.device)
        """
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

    def compute_ijcai_contrastive_loss(self, motion_for_blob, features_for_blob, first_foa_row_col, saccade_bool,
                                       prev_features_for_blob):

        foa_blob = self.__compute_foa_moving_blob(motion_for_blob, first_foa_row_col,
                                                  motion_threshold=self.options['motion_threshold'])
        # getting the activation associated to the focus-of-attention-coordinates at every layer
        activations_foa = features_for_blob[None, 0, :, first_foa_row_col[0], first_foa_row_col[1], None]

        if self.__fully_online:
            activations_foa_prev = \
                self.__activations_foa_prev if self.__activations_foa_prev is not None else activations_foa.detach()
        else:
            # gather the motion values in the foa location
            motion_in_foa_location = motion_for_blob[0, :, first_foa_row_col[0], first_foa_row_col[1]]
            # backward warp of the foa location to the previous frame ones
            prev_first_foa_row_col = (first_foa_row_col - motion_in_foa_location).to(torch.long)
            # get features in previous foa location # CAN BE IMPROVED; maybe take the real v?
            activations_foa_prev = prev_features_for_blob[None, 0, :, prev_first_foa_row_col[0],
                                   prev_first_foa_row_col[1], None]

        ijcai_dic = self.net.compute_unsupervised_loss(features_for_blob,
                                                       # TODO old_features, new_features, formate_required?
                                                       activations_foa if not saccade_bool else None,
                                                       activations_foa_prev if not saccade_bool else None,
                                                       None,
                                                       None,
                                                       None,
                                                       None,
                                                       foa_blob,
                                                       first_foa_row_col)

        ############################# SIMPLIFICATIONNN  #############################
        """

        # computing the motion-based 'blob' around the focus of attention
        foa_blob = self.__compute_foa_moving_blob(motion_for_blob, foa_row_col,
                                                  motion_threshold=self.options['motion_threshold'])

        b, f, hw = features_for_blob.shape[0], features_for_blob.shape[1], self.h * self.w
        # getting the activation associated to the focus-of-attention-coordinates at every layer
        foa_frontier = foa_row_col[:, 0, None] * self.w + foa_row_col[:, 1, None]  # [b, 1] flattened coordinates of the FOA
        foa_frontier = foa_frontier.unsqueeze(1)  # get all the features in that foa coordinate
        activations_foa = torch.gather(features_for_blob.view(b, f, hw), dim=2, index=foa_frontier.expand(b, f, 1))

        # previous activations on the foa coordinates (if available)
        if self.__fully_online:
            activations_foa_prev = \
                self.__activations_foa_prev if self.__activations_foa_prev is not None else activations_foa.detach()
        else:
            # BORDELLO
            # get it from old_frame activatations...by backward-warping the coordinates with motion
            old_coordinates_row = foa_row_col[:, 0] - motion_for_blob[:, 0, foa_row_col[0], foa_row_col[1]]
            old_coordinates_col = foa_row_col[:, 0] - motion_for_blob[:, 1, foa_row_col[0], foa_row_col[1]]
            activations_foa_prev = old_features_for_blob[:, old_coordinates_row, old_coordinates_col]


        ijcai_dic = self.net.compute_unsupervised_loss(features_for_blob,   # TODO old_features, new_features, formate_required?
                                                    activations_foa if not saccade_bool else None,
                                                    activations_foa_prev if not saccade_bool else None,
                                                    None,
                                                    None,
                                                    None,
                                                    None,
                                                    foa_blob,
                                                    foa_row_col)   
        """

        ijcai_loss, details = ijcai_dic
        loss_s_in, loss_s_out, loss_t, loss_mi, cond_entropy, entropy, mi, points_indices_in, points_indices_out = details

        sampled_points = np.zeros((self.w * self.h, 3), dtype=np.uint8)
        if points_indices_in is not None:
            points_indices_in = points_indices_in.cpu().numpy()
            sampled_points[points_indices_in, 0] = 255
            sampled_points[points_indices_in, 1] = 0
            sampled_points[points_indices_in, 2] = 0
        if points_indices_out is not None:
            points_indices_out = points_indices_out.cpu().numpy()
            sampled_points[points_indices_out, 0] = 0
            sampled_points[points_indices_out, 1] = 0
            sampled_points[points_indices_out, 2] = 255
        sampled_points = sampled_points.reshape(self.h, self.w, 3)
        sampled_points = cv2.blur(sampled_points, (5, 5))
        sampled_points = sampled_points.reshape(self.h * self.w, 3)
        sampled_points[sampled_points[:, 0] > 0, 0] = 255
        sampled_points[sampled_points[:, 1] > 0, 1] = 255
        sampled_points[sampled_points[:, 2] > 0, 2] = 255
        sampled_points = sampled_points.reshape(self.h, self.w, 3)

        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        if self.__fully_online:
            self.__activations_foa_prev = activations_foa.detach()  # TODO gestire batch + shuffle!

        self.__stats.update({
            "loss_s_in": loss_s_in,
            "loss_s_out": loss_s_out,
            "loss_t": loss_t,
        })
        foa_blob = lve.utils.torch_float_01_to_np_uint8(foa_blob.to(torch.float))
        self.add_outputs({
            "blob": foa_blob,  # image
            "sampled-points": sampled_points,  # image
        })

        self.__unsup_loss = self.__unsup_loss + ijcai_loss
