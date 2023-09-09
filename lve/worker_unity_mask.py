import os
import numpy as np
import random
from random import randint, uniform
import lve
import torch
import cv2
import torchvision
from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as tF
import torch.nn.functional as F
import time
from collections import OrderedDict
import bisect


class WorkerUnityMask(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this
        self.device = torch.device(options["device"] if "device" in options else "cpu")  # device
        self.b = self.options["batch_size"]
        self.__w, self.__h = w, h

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # registering supported commands
        self.register_command("reset_foa", self.__handle_command_reset_foa)

        # model parameters
        self.rho = self.options["rho"]

        self.net_options = self.options["net"]

        # processors
        self.blur = lve.BlurCV(self.w, self.h, self.c, self.device)
        self.optical_flow = lve.OpticalFlowCV()
        self.geymol = lve.GEymol(self.options["foa"], self.device)

        # pretrained model for instance segmentation
        self.__net_backbone_mask()

        # some model parameters
        self.__first_frame = True

        # misc
        self.__saccade = False

        # misc (data about the whole worker to print on screen or save to disk)
        self.__stats = OrderedDict([('rho', self.rho), ('threshold', self.net_options["threshold"])])

    def __net_backbone_mask(self):

        self.category_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        sup_dictionary = dict(enumerate(self.category_names))
        sup_dictionary_inv = {}
        for k, v in sup_dictionary.items():
            sup_dictionary_inv[v] = k
        self.augment_supervision_map(sup_dictionary_inv, max_classes=len(self.category_names))

        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.mask_rcnn.to(self.device)  # sending model to device
        self.mask_rcnn.eval()

        # created one time into the correct device (to avoid transfer for each inference step)
        prediction_tensor = torch.zeros(
            [self.b, len(self.category_names), self.__h, self.__w], dtype=torch.float32)

        self.prediction_tensor = prediction_tensor.to(self.device)

    def process_frame(self, frame, of=None, supervisions=None, foa=None):

        # blur, motion, focus of attention
        self.b = len(frame)

        self.__frames, motions_or_foas, _saccades, _blurred, _foas, list_frames = self.__compute_sequential_ops_and_batch(
            frame, of)

        pred_classes = self.__net_inference_mask(list_frames, threshold=self.net_options["threshold"])

        self.__stats = OrderedDict([('rho', self.rho), ('threshold', self.net_options["threshold"])])

        for i in range(0, self.b):
            self.add_outputs({"motion": of[i],  # binary
                              "blurred": _blurred[i],  # PNG image
                              "stats.foa": {"x": _foas[i][0], "y": _foas[i][1], "vx": _foas[i][2], "vy": _foas[i][3],
                                            "saccade": _saccades[i - 1] if i > 0 else self.__saccade},
                              "stats.worker": self.__stats,
                              "logs.worker": list(self.__stats.values()),  # CSV log
                              "tb.worker": self.__stats,
                              "sup.targets": supervisions[i][0],
                              "sup.indices": supervisions[i][1]}, batch_index=i)  # tensorboard
            if self.heavy_output_data_needed:
                self.add_outputs({"predictions": pred_classes[i, None].detach().cpu().numpy()},
                                 batch_index=i)  # binary

                #
                # self.add_outputs({"targets": pred_classes[i, None].detach().cpu().numpy()},
                #                  batch_index=i)  # binary

    def update_model_parameters(self):
        pass

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading weights
        # self.dummy_weights = np.load(worker_model_folder + "dummy_weights.npz", allow_pickle=True)['arr_0']

        # loading other parameters
        params = lve.utils.load_json(worker_model_folder + "params.json")

        # setting up the internal elements using the loaded parameters
        self.rho = params["rho"]
        self.geymol.reset(params["foa_y"], params["foa_t"])
        self.geymol.first_call = False

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving weights
        # np.savez_compressed(worker_model_folder + "dummy_weights.npz", self.dummy_weights)

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "params.json", {"rho": self.rho,
                                                                  "threshold": self.net_options["threshold"],
                                                                  "supervision_map": self.get_supervision_map()})

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "blurred": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "stats.foa": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys()),  # first line of CSV
            # "sup": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "sup.indices": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.targets": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.map": {'data_type': lve.OutputType.JSON, 'per_frame': False}

        }

        output_types.update({"predictions": {'data_type': lve.OutputType.BINARY, 'per_frame': True}})
        # output_types.update({"probabilities": {'data_type': lve.OutputType.BINARY, 'per_frame': True}})

        return output_types

    def print_info(self):
        print("   {rho: " + str(self.rho) + ", eta: " + str(self.options["eta"]) + "}")

    def __handle_command_reset_foa(self, command_value, batch_index=0):
        self.geymol.reset([command_value['y'], command_value['x'],
                           2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1)),
                           2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))])

    def __transform_frame_mask(self):
        self.transformed_frame = T.Compose([T.ToTensor()])

    def __transform_frame_seg(self, frames):

        trf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        for i in range(self.b):
            frames[i] = trf(frames[i])
        return F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)

    def __compute_sequential_ops_and_batch(self, batch_frames_np_uint8, batch_motion_np_float32):
        batch_frames = [None] * self.b
        batch_mask = [None] * self.b
        batch_motion_or_foa = [None] * self.b
        _batch_saccades = [None] * self.b
        _batch_foa_np_float64 = [None] * self.b
        _batch_blurred_frames_np_uint8 = [None] * self.b

        # sequential operations on the batched data
        for i in range(0, self.b):

            # blurring factor
            if self.rho < 1.0 and (not self.__first_frame or i > 0):
                diff_rho = 1.0 - self.rho
                self.rho = self.rho + self.options["eta"] * diff_rho  # eta: hot-changeable option
                if self.rho > 0.99:
                    self.rho = 1.0

            # blurring
            frame_np_uint8 = self.blur(batch_frames_np_uint8[i], blur_factor=1.0 - self.rho).astype(np.uint8)
            frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8, device=self.device)
            _batch_blurred_frames_np_uint8[i] = frame_np_uint8

            # grayscale-instance of the (blurred) input frame
            if not self.frame_is_gray_scale:
                frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
                frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8, device=self.device)
            else:
                frame_gray_np_uint8 = frame_np_uint8
                frame_gray = frame

            # optical flow
            if batch_motion_np_float32 is None or batch_motion_np_float32[i] is None:
                motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

                if batch_motion_np_float32 is not None:
                    batch_motion_np_float32[i] = motion_np_float32  # updating
            else:
                motion_np_float32 = batch_motion_np_float32[i]  # h x w x 2
                motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

            # focus of attention
            foa, next_will_be_fixation = self.geymol.next_location(frame_gray, motion,
                                                                   frame_gray_uint8_cpu=frame_gray_np_uint8)

            # storing references
            batch_frames[i] = frame
            batch_mask[i] = torch.squeeze(frame)
            if not self.net_options["foa_coherence"]:
                batch_motion_or_foa[i] = motion
            else:
                batch_motion_or_foa[i] = torch.from_numpy(foa[0:2].astype(np.long)).to(self.device).view(1, 2)
            _batch_saccades[i] = not next_will_be_fixation
            _batch_foa_np_float64[i] = foa

        frames = torch.cat(batch_frames, dim=0)  # TODO returned also this here to have a list of tensirs
        motions_or_foas = torch.cat(batch_motion_or_foa, dim=0)

        return frames, motions_or_foas, _batch_saccades, _batch_blurred_frames_np_uint8, _batch_foa_np_float64, batch_mask

    def __net_inference_mask(self, frames, threshold):
        # pre-processing  maybe already done
        # frames = self.transformed_frame(frames)  # even with batched data
        pred = self.mask_rcnn(frames)

        # reset the prediction tensor
        self.prediction_tensor.detach_()
        self.prediction_tensor.zero_()

        for j in range(self.b):
            pred_score = pred[j]['scores'] # various scores of detected objects
            indices_gt = pred_score > threshold # get scores greater than threshold (dynamic) => boolean, where position
            labels_t = pred[j]["labels"][indices_gt]  # labels of detected  objects => used as index
            # now select the masks corresponding to predicted objects, reshape in n_masks x h x w
            mask_t = pred[j]["masks"][indices_gt].view(indices_gt.sum(), self.__h, self.__w)
            mask_t_new = torch.zeros_like(mask_t)
            mask_t_new.masked_fill_(mask_t > 0.5, 1.)  # clip values, where mask_t is greater than 0.5 insert 1
            # now insert pred_score value in all the mask => using broadcasting, create pred_score with shape:
            # n_dect x 1 x 1
            mask_t_new *= pred_score[indices_gt].view(mask_t_new.shape[0], 1, 1)
            # prediction tensor has dims: b x 91 x h x w => insert in all postions indexed by labels_t
            # (also if are the same, e.g. two persons in different positions) the values of masks
            self.prediction_tensor[j].index_put_((labels_t,), mask_t_new, accumulate=True)
            # put one where sum is zero in class 0=> background
            self.prediction_tensor[j, 0, torch.sum(self.prediction_tensor[j], dim=0) == 0.] = 1.0

        return self.prediction_tensor
