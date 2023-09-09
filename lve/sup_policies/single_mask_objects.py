import numpy as np
from lve.sup_policies.sup_policy import SupervisionPolicy


class SingleMaskObjectsPolicy(SupervisionPolicy):

    def _apply(self, batch_frame, batch_of, batch_supervisions, batch_foa):
        filtered_supervisions = []

        for frame, supervisions in zip(batch_frame, batch_supervisions):
            frame_size = frame.shape[:-1]
            reconstr_sup = np.full(frame_size, 255).flatten()
            sup_targets, sup_indices = supervisions
            reconstr_sup[sup_indices] = sup_targets
            sup_targets = reconstr_sup.reshape(frame_size)

            # indices = [0,1,2,3,4,5]
            # targets = [0,0,1,1,1,0]
            # compute pixels in the mask
            of_not_null_idx = np.argwhere(sup_targets==0)

            if of_not_null_idx.size == 0:
                filtered_supervisions.append([None, None])  # Stefano
                continue # return no supervision when no object is moving

            # assume only one object move at a time, so we can take the class at pixel "of_not_null_idx[0]"
            target = sup_targets[of_not_null_idx[0][0], of_not_null_idx[0][1]]

            centroid = of_not_null_idx.mean(axis=0).round().astype(int)
            centroid_class = sup_targets[centroid[0], centroid[1]]

            filter_sup_targets = [target]

            if centroid_class == target and False:
                filter_sup_index = [centroid[0]*frame_size[1] + centroid[1]]  # convert to flat representation
            else:
                distances = np.linalg.norm(of_not_null_idx - centroid, axis=1)
                closest_to_centroid = of_not_null_idx[np.argmin(distances)]
                filter_sup_index = [closest_to_centroid[0]*frame_size[1] + closest_to_centroid[1]]

            filtered_supervisions.append([filter_sup_targets, filter_sup_index])

        return filtered_supervisions
