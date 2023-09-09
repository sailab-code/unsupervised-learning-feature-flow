import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from confeat.network import WholeNet, StandardFBlock, StandardDBlock, FromDiskDBlock


class BaseOpenSetClassifier(nn.Module):

    def __init__(self, options, sup_buffer, device, distance='euclidean'):
        super(BaseOpenSetClassifier, self).__init__()
        self.w = options["w"]
        self.h = options["h"]
        self.sup_buffer = sup_buffer
        self.options = options
        self.device = device
        self.distance = distance
        if type(self.options["dist_threshold"]) != list:
            self.thresh_list = [self.options["dist_threshold"]]
        else:
            self.thresh_list = self.options["dist_threshold"]

    def forward(self, frame_embeddings):
        pass

    def forward_and_compute_supervised_loss(self, data, labels):
        pass

    def predict(self, frame_embeddings):
        pass

    def compute_mask_of_pixels_to_predict(self, frame_embeddings):
        b = frame_embeddings.shape[0]  # b x num-pixels x dim
        template_list = self.sup_buffer.get_embeddings()  # list of embedding vector (each of them 1 x dim)
        _, template_classes = self.sup_buffer.get_embeddings_labels()
        #   frame_embeddings = [ num-pixels, num_what,]
        if len(template_list) > 0:
            dists_from_templates = [None] * len(template_list)
            for i in range(len(template_list)):
                template = template_list[i]  # [1, num_what ]
                if self.distance == '1-dot':
                    # dists_from_templates[i] = 1. - torch.matmul(frame_embeddings, template.t())  # batched   # TODO solve BUG! output is [b, hw, 1]
                    dists_from_templates[i] = 1. - torch.matmul(frame_embeddings, template.squeeze().t())  # batched   # removed the additional dim
                    # dists_from_templates[i] = 1. - torch.sum(frame_embeddings * template, dim=1)
                    # dists_from_templates[i] = 2. - (torch.inner(frame_embeddings, template)).squeeze()
                elif self.distance == 'euclidean':
                    dists_from_templates[i] = torch.sum(torch.pow(frame_embeddings - template.unsqueeze(0), 2.0),
                                                        dim=2)  # batch_size x num-pixels
                else:
                    raise NotImplementedError

            dists_from_templates = torch.stack(dists_from_templates, dim=1)  # [batch_size, num-templates, num-pixels]
            min_dists_from_templates, min_indexes_from_template = torch.min(dists_from_templates,
                                                                            dim=1)  # batch_size x num-pixels

            mask_list = []  # list of batch_size x num-pixels (it is list due to the multiple thresholds)

            for thresh in self.thresh_list:
                mask_list.append(min_dists_from_templates <= thresh)

            return mask_list, min_dists_from_templates, \
                   template_classes[min_indexes_from_template].to(frame_embeddings.device)  # batch_size x num-pixels

        else:
            return [torch.zeros((b, self.h * self.w), dtype=torch.bool, device=frame_embeddings.device)] * len(
                self.thresh_list), None, None


class NNOpenSetClassifier(BaseOpenSetClassifier):

    def __init__(self, options, sup_buffer, device, distance='euclidean'):
        super(NNOpenSetClassifier, self).__init__(options, sup_buffer, device, distance)

    def predict(self, frame_embeddings, frame_embeddings_alt=None):
        # frame_embeddings [batch_size, num_pixel, num_what]
        b = frame_embeddings.shape[0]

        self.thresh_list = list(self.options["dist_threshold"])
        mask_list, _, neigh_classes = self.compute_mask_of_pixels_to_predict(frame_embeddings)
        if self.options['decoder_input_norm'] == 'both':
            mask_list_alt, _, neigh_classes_alt = self.compute_mask_of_pixels_to_predict(frame_embeddings_alt)

        if len(self.sup_buffer.get_embeddings()) == 0:
            masked_pred_list_def = [torch.zeros((b, self.h * self.w, self.options["supervised_categories"]),
                                device=frame_embeddings.device)] * len(self.thresh_list * (2 if self.options['decoder_input_norm'] == 'both' else 1))
            neigh_classes_list_def = [(self.options["supervised_categories"] - 1) * torch.ones((b, self.h * self.w),
                                                                             device=frame_embeddings.device)]\
                                     * len(self.thresh_list * (2 if self.options['decoder_input_norm'] == 'both' else 1))
            return masked_pred_list_def, \
                   mask_list + (mask_list_alt if self.options['decoder_input_norm'] == 'both' else []), \
                   neigh_classes_list_def, torch.zeros((b, self.h * self.w), device=frame_embeddings.device)
        else:
            def compute_pred_neigh_lists(masks, preds, ncat):
                one_hot_classes = \
                    torch.nn.functional.one_hot(preds,num_classes=ncat).to(torch.float)

                masked_pred_list = []
                neigh_classes_list = []
                for mask in masks:
                    neigh_classes_temp = preds.detach().clone()
                    masked_pred_list.append(one_hot_classes * mask.view(b, self.w * self.h, 1))
                    neigh_classes_temp[mask == False] = ncat - 1
                    neigh_classes_list.append(neigh_classes_temp)
                return masked_pred_list, neigh_classes_list

            masked_pred_list, neigh_classes_list = compute_pred_neigh_lists(mask_list, neigh_classes, ncat=self.options["supervised_categories"])
            if self.options['decoder_input_norm'] == 'both':
                masked_pred_list_alt, neigh_classes_list_alt = compute_pred_neigh_lists(mask_list_alt, neigh_classes_alt, ncat=self.options["supervised_categories"])
                masked_pred_list += masked_pred_list_alt
                neigh_classes_list += neigh_classes_list_alt
                mask_list += mask_list_alt

            # other stuff
            unmasked_neigh_classes = neigh_classes.detach().clone()
            return masked_pred_list, mask_list, neigh_classes_list, unmasked_neigh_classes

    def forward_and_compute_supervised_loss(self, data, labels):
        return torch.tensor(0.0, device=self.device), 0.0


class NeuralOpenSetClassifier(BaseOpenSetClassifier):

    def __init__(self, options, sup_buffer, device):
        super(NeuralOpenSetClassifier, self).__init__(options, sup_buffer, device)

        self.supervised_projection = nn.Sequential(
            nn.Linear(in_features=options['num_what'], out_features=100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=options['supervised_categories'], bias=True))

    def forward(self, frame_embeddings):
        return self.supervised_projection(frame_embeddings) if frame_embeddings is not None else None

    def forward_and_compute_supervised_loss(self, embeddings, labels):
        logits = self.forward(embeddings)
        if logits is None:
            return torch.tensor(0., device=self.device), 0.

        lambda_l = self.options['lambda_l']
        loss_sup = F.cross_entropy(logits, labels, reduction='mean')

        loss = lambda_l * loss_sup
        return loss, loss_sup.item()

    def predict(self, frame_embeddings):
        logits = self.supervised_projection(frame_embeddings)
        mask, _, _ = self.compute_mask_of_pixels_to_predict(frame_embeddings)
        supervised_probs = torch.softmax(logits, dim=1) * mask.view(self.w * self.h, 1)

        return supervised_probs, mask, torch.argmax(supervised_probs, dim=1)


class NetConj(nn.Module):

    def __init__(self, options, device, sup_buffer):
        super(NetConj, self).__init__()

        # keeping track of the network options
        self.options = options
        self.device = device
        # if we are dealing with a pretrained model, load directly the net

        if self.options["arch_mode"] == "pretrained":
            self.whole_model = options["_whole_net"].to(device)
        else:
            self.whole_model = WholeNet(options, device).to(device)

        self.w = options["w"]
        self.h = options["h"]

        self.__old_frames = None
        self.num_features = options["total_features"]

        # supervised module
        self.classifier = None
        if options["classifier"] == "neural":
            self.classifier = NeuralOpenSetClassifier(options, sup_buffer, self.device)
        elif options["classifier"] == "NN":
            distance = "1-dot" if self.options['vision_block']['features']['normalize'] else "euclidean"
            self.classifier = NNOpenSetClassifier(options, sup_buffer, self.device, distance=distance)
        else:
            raise NotImplementedError


    def init_weights(self, m):
        value = self.options['vision_block']["weights_reinit"]
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=value)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=value)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, frame, old_frame, piggyback_frames, motion_disk=None, augmented_count=0):
        format_required_raw_features = None
        b = frame.shape[0]
        b_real = int(b // int(augmented_count + 1))

        # notice that each returned variable is a list, one element for each block
        # unnormalized scores
        displacements, features_current, features_old, lower_features_current, lower_features_old,\
            logits_current, logits_old = self.whole_model(frame, old_frame, motion_disk)

        def normalize_features(x):
            return x / (torch.norm(x, dim=1, keepdim=True) + 1e-12)

        format_required_features_alt = None

        # prepare input for openset classifier (concatenated or last layer outputs) - notice: if need to normalized, the
        # concatened representation is normalized another times
        # what is done below only consider the current frame and not the piggyback ones
        if self.options['decoder_input'] == 'concatenated':
            # to keep the same size - select first element (not the microsaccade)
            if self.options['vision_block']['features']['normalize']:
                if self.options['decoder_input_norm'] == 'separated':
                    format_required_features = torch.cat([normalize_features(f) for f in features_current], 1)[0:b_real]
                    format_required_old_features = torch.cat([normalize_features(f) for f in features_old], 1)[0:b_real]  # to keep the same size
                elif self.options['decoder_input_norm'] == 'both':
                    format_required_features = torch.cat(features_current, 1)[0:b_real]
                    format_required_old_features = torch.cat(features_old, 1)[0:b_real]  # to keep the same size

                    format_required_features_alt = torch.cat([normalize_features(f) for f in features_current], 1)[0:b_real]
                else: # standard
                    format_required_features = torch.cat(features_current, 1)[0:b_real]
                    format_required_old_features = torch.cat(features_old, 1)[0:b_real]  # to keep the same size
            else:
                format_required_features = torch.cat(features_current, 1)[0:b_real]
                format_required_old_features = torch.cat(features_old, 1)[0:b_real]  # to keep the same size
        elif self.options['decoder_input'] == 'last-block' or self.options['decoder_input'] == 'per-block':
            format_required_features = features_current[-1][0:b_real]  # to keep the same size
            format_required_old_features = features_old[-1][0:b_real]  # to keep the same size
        else:
            raise NotImplementedError

        if self.options['vision_block']['features']['normalize']:
            format_required_features = normalize_features(format_required_features)
            format_required_old_features = normalize_features(format_required_old_features)
            if format_required_features_alt is not None:
                format_required_features_alt = normalize_features(format_required_features_alt)


        # update piggy embeddings
        if piggyback_frames is not None:
            # update embeddings of piggyback, still as a list
            piggy_list = []
            inp = piggyback_frames
            for i, module in enumerate(self.whole_model.vision_blocks):
                # use the internal feature block to update the embeddings only
                emb = module.features_block(inp)
                inp = emb
                piggy_list.append(emb)
            if self.options['decoder_input'] == 'concatenated':
                format_required_piggyback = torch.cat(piggy_list, 1)
            elif self.options['decoder_input'] == 'last-block' or self.options['decoder_input'] == 'per-block':
                format_required_piggyback = piggy_list[-1]
            if self.options['vision_block']['features']['normalize']:
                format_required_piggyback = format_required_piggyback / (
                        torch.norm(format_required_piggyback, dim=1, keepdim=True) + 1e-12)
        else:
            format_required_piggyback = torch.empty(0)

        format_required_features_reshaped = format_required_features.view(b_real, self.num_features, -1).permute(0, 2, 1)
        if format_required_features_alt is not None:
            format_required_features_alt_reshaped = format_required_features_alt.view(b_real, self.num_features, -1).permute(0, 2, 1)
        else:
            format_required_features_alt_reshaped = None

        # masked_supervised_probs: list with class prediction (masked), each (1, npix, nclass)
        # mask: list with masks
        # prediction_idx: list with predicted class labels (1, npix) -> one item per threhsold
        # unmasked_prediction_idx: predicted class labels (1, npix)
        masked_supervised_probs, mask, prediction_idx, unmasked_prediction_idx = \
            self.classifier.predict(format_required_features_reshaped, format_required_features_alt_reshaped)

        if format_required_raw_features is None: format_required_raw_features = format_required_features

        return features_current, features_old, lower_features_current, lower_features_old, format_required_raw_features, format_required_features, \
               format_required_piggyback, displacements, masked_supervised_probs, mask, prediction_idx, \
               unmasked_prediction_idx, format_required_old_features, logits_current, logits_old

    def compute_unsupervised_loss(self, whats, activations_foa, activations_foa_prev,
                                  unsupervised_probs, avg_unsupervised_probs, unsupervised_probs_foa_blob,
                                  unsupervised_probs_foa, foa_blob, foa_row_col):

        lambda_s_in = self.options['lambda_s_in']
        lambda_s_out = self.options['lambda_s_out']
        lambda_t = self.options['lambda_t']

        normalize = self.options['vision_block']['features']['normalize']

        # spatial coherence loss
        loss_s_in, loss_s_out, points_indices_in, points_indices_out = \
            self.__compute_spatial_coherence(whats, foa_blob,
                                             foa_row_col, contrastive=lambda_s_out > 0.,
                                             normalized_data=normalize)

        # temporal coherence loss
        if activations_foa is not None and activations_foa_prev is not None:
            if normalize:
                loss_t = (1. - torch.sum(activations_foa * activations_foa_prev)) \
                    if activations_foa_prev is not None else torch.tensor(0., device=self.device)
            else:
                loss_t = torch.sum((activations_foa - activations_foa_prev) ** 2) \
                    if activations_foa_prev is not None else torch.tensor(0., device=self.device)
        else:
            loss_t = torch.tensor(0.0, device=self.device)

        # mutual information (squared approximation)
        if unsupervised_probs is not None:
            lambda_c = self.options['lambda_c']
            lambda_e = self.options['lambda_e']
            cond_entropy = -torch.sum(torch.pow(unsupervised_probs, 2)).div(unsupervised_probs.shape[0])
            entropy = -torch.sum(torch.pow(avg_unsupervised_probs, 2))
            loss_mi = -(lambda_e * entropy - lambda_c * cond_entropy)

            # spatial coherence loss on the unsupervised symbols
            if unsupervised_probs_foa_blob is not None and unsupervised_probs_foa is not None:
                loss_smi = torch.sum((unsupervised_probs_foa_blob - unsupervised_probs_foa) ** 2)  # broadcast
            else:
                loss_smi = torch.tensor(0.0, device=self.device)

            # mutual information: useful values to be returned
            pm = unsupervised_probs.shape[1]
            mi = (entropy.item() - cond_entropy.item()) * pm / (pm - 1.)  # [0,1]
            cond_entropy = (cond_entropy.item() + 1.) * (pm / (pm - 1.))  # [0,1]
            entropy = (entropy.item() + 1.) * (pm / (pm - 1.))  # [0,1]
        else:
            cond_entropy = entropy = mi = -1.0
            loss_mi = torch.tensor(0.0, device=self.device)
            loss_smi = torch.tensor(0.0, device=self.device)

        # final loss
        loss = lambda_s_in * loss_s_in + lambda_s_out * loss_s_out + lambda_t * loss_t \
               + loss_mi + lambda_s_in * loss_smi
        return loss, (loss_s_in.item(), loss_s_out.item(), loss_t.item(), loss_mi.item(), cond_entropy, entropy, mi,
                      points_indices_in, points_indices_out)

    def compute_supervised_loss(self, embeddings, labels):
        return self.classifier.forward_and_compute_supervised_loss(embeddings, labels)

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

    def print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")

    # def __sample_points_in_foa_blob(self, n_in, foa_blob, foa_row_col, ensure_foa_is_there=False):
    #     assert n_in >= 1, "At least two points must be sampled."
    #     b = foa_blob.shape[0]
    #     all_indices = torch.arange(b * self.w * self.h, device=foa_blob.device)  # b * w*h
    #     foa_blob_indices = all_indices[foa_blob.view(-1)]  # NOT batch-friendly??
    #
    #     if foa_blob_indices.shape[0] > 0:
    #         perm = torch.randperm(foa_blob_indices.shape[0])
    #         sampled_indices = foa_blob_indices[perm[:n_in]]
    #
    #         if ensure_foa_is_there: # TODO!
    #             foa_index = foa_row_col[:, 0, None] * self.w + foa_row_col[:, 1, None]  # TODO!
    #             if foa_index not in sampled_indices:
    #                 sampled_indices[0] = foa_index
    #
    #         return sampled_indices
    #     else:
    #         return None

    def __sample_points_in_foa_blob(self, n_in, foa_blob, foa_row_col, ensure_foa_is_there=True):
        assert n_in >= 1, "At least two points must be sampled."

        all_indices = torch.arange(self.w * self.h, device=foa_blob.device)
        foa_blob_indices = all_indices[foa_blob.view(-1)]

        if foa_blob_indices.shape[0] > 0:
            perm = torch.randperm(foa_blob_indices.shape[0])
            sampled_indices = foa_blob_indices[perm[:n_in]]

            if ensure_foa_is_there:
                foa_index = foa_row_col[0] * self.w + foa_row_col[1]
                if foa_index not in sampled_indices:
                    sampled_indices[0] = foa_index

            return sampled_indices
        else:
            return None

    def __sample_points_out_of_foa_blob(self, n_out, foa_blob, foa_row_col, spread_factor):
        assert n_out >= 1, "At least one point must be sampled."

        # generating gaussian distribution around focus of attention
        sqrt_area = math.sqrt(torch.sum(foa_blob))
        gaussian_foa = torch.distributions.normal.Normal(torch.tensor([float(foa_row_col[0]), float(foa_row_col[1])]),
                                                         torch.tensor([spread_factor * sqrt_area,
                                                                       spread_factor * sqrt_area]))

        # sampling "n_out" points out of the foa blob
        got_n_out = 0
        points_indices_per_attempt = []

        attempts = 0
        max_attempts = 10

        while got_n_out < n_out and attempts < max_attempts:

            # sampling
            points_row_col = gaussian_foa.sample(torch.Size([n_out - got_n_out])).to(torch.long)

            # avoiding out-of-bound points
            points_row_col = points_row_col[torch.logical_and(
                torch.logical_and(points_row_col[:, 0] >= 0, points_row_col[:, 0] < self.h),
                torch.logical_and(points_row_col[:, 1] >= 0, points_row_col[:, 1] < self.w)), :]

            if points_row_col.shape[0] > 0:
                foa_blob = foa_blob.view(-1)

                # converting to indices
                points_indices = (points_row_col[:, 0] * self.w + points_row_col[:, 1]).to(foa_blob.device)

                # keeping only indices out of the foa_blob
                points_indices = points_indices[~foa_blob[points_indices]]

                # appending the found points to an ad-hoc list
                if points_indices.shape[0] > 0:
                    points_indices_per_attempt.append(points_indices)
                    got_n_out += points_indices.shape[0]

            # counting the number of sampling attempts done so far
            attempts += 1

        # from list to tensor
        if len(points_indices_per_attempt) > 0:
            points_indices_out = torch.cat(points_indices_per_attempt)
        else:
            points_indices_out = None

        return points_indices_out

    def __compute_spatial_coherence(self, full_frame_activations, foa_blob, foa_row_col,
                                    contrastive=True, normalized_data=False):

        num_pairs = self.options['num_pairs']
        if self.options['num_pairs'] < 0:
            n_in = torch.sum(foa_blob)
            if n_in == 0: n_in = 1
            points_indices_in = torch.arange(end=foa_blob.nelement())[foa_blob.flatten()]
        else:
            # in order to get "num_pairs" pairs insider the foa blob,
            # we need to sample (1 + sqrt(1 + 8*num_pairs)) * 0.5 nodes
            n_in = int(
                (1. + math.sqrt(1 + 8 * num_pairs)) * 0.5)  # inside the foa blob (it will yield "num_pairs" edges)

            # sampling
            points_indices_in = self.__sample_points_in_foa_blob(n_in, foa_blob, foa_row_col, ensure_foa_is_there=True)

            # checking

        if points_indices_in is None or points_indices_in.nelement() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None, None
        # getting activations
        full_frame_activations = full_frame_activations.view(full_frame_activations.shape[0],
                                                             full_frame_activations.shape[1],
                                                             -1)
        A_in = full_frame_activations[0, :, points_indices_in].t()

        # computing loss
        if not normalized_data:
            A_in_diff = A_in[:, None] - A_in  # difference between all the pairs

            loss_s_in = torch.sum(A_in_diff * A_in_diff) / (n_in * (n_in - 1))
        else:
            loss_s_in = 1.0 - ((torch.sum(torch.matmul(A_in, A_in.t())) - n_in) / (n_in * (n_in - 1)))

        if contrastive:
            if self.options['num_pairs'] < 0:
                points_indices_out = torch.arange(end=foa_blob.nelement())[~foa_blob.flatten()]
            else:
                # in order to get "num_pairs" pairs composed of one element inside the blob and one outside,
                # we need to sample num_pairs / (num_points_inside_foa_blob + 1) points outside the foa blob
                n_out = int(num_pairs / (n_in + 1.))  # outside the foa blob (it will yield "num_pairs" edges)

                # sampling
                points_indices_out = self.__sample_points_out_of_foa_blob(n_out, foa_blob, foa_row_col,
                                                                          spread_factor=self.options["spread_factor"])

                # checking
                if points_indices_out is None:
                    return loss_s_in, torch.tensor(0.0, device=self.device), points_indices_in, None

            # getting activations
            A_out = full_frame_activations[0, :, points_indices_out]

            # computing loss
            if not normalized_data:
                loss_s_out = 1.0 / (torch.mean(torch.sum(A_in * A_in, dim=1, keepdim=True)
                                               - 2.0 * torch.matmul(A_in, A_out)
                                               + torch.sum(A_out * A_out, dim=0, keepdim=True)) + 1e-20)
            else:
                loss_s_out = 1.0 + torch.mean(torch.matmul(A_in, A_out))

            return loss_s_in, loss_s_out, points_indices_in, points_indices_out
        else:
            return loss_s_in, torch.tensor(0., device=self.device), points_indices_in, None
