import torch


def mi(p, avg_p=None, weights=None):
    p = torch.max(p, 0.00001)
    log_p = torch.log(p).div_(torch.log(p.shape[1]))  # log base: num-features
    cond_entropy = -torch.sum(torch.mean(p * log_p, dim=(0, 2, 3)))

    avg_p = torch.mean(p, dim=(0, 2, 3)) if avg_p is None else avg_p
    log_avg_p = torch.log(avg_p).div_(torch.log(p.shape[1]))  # log base: num-features
    entropy = -torch.sum(avg_p * log_avg_p)

    if weights is not None:
        weighed_mi = weights[1] * entropy - weights[0] * cond_entropy
    else:
        weighed_mi = entropy - cond_entropy

    return weighed_mi, cond_entropy, entropy


def mi_approx(p, avg_p=None, weights=None):
    cond_entropy = -torch.sum(torch.pow(p, 2)).div(p.shape[0] * p.shape[2] * p.shape[3])
    avg_p = torch.mean(p, dim=(0, 2, 3)) if avg_p is None else avg_p
    entropy = -torch.sum(torch.pow(avg_p, 2))

    if weights is not None:
        weighed_mi = weights[1] * entropy - weights[0] * cond_entropy
    else:
        weighed_mi = entropy - cond_entropy

    return weighed_mi, cond_entropy, entropy


def coherence(prev, cur, indices=None, delta=1.0):
    if indices is not None:
        prev = swap_by_indices(prev, indices)

    diff = prev - cur
    return torch.dot(diff.view(-1), diff.view(-1)) / (delta**2)


def create_image_plane_coords(b, h, w, device=None):
    if device is None:
        device = torch.device("cpu")
    index_i = torch.arange(0, h, dtype=torch.float32, device=device).repeat([w, 1]).t().view(1, h, w)
    index_j = torch.arange(0, w, dtype=torch.float32, device=device).repeat([h, 1]).view(1, h, w)
    if b > 1:
        index_i = index_i.expand([b, -1, -1])
        index_j = index_j.expand([b, -1, -1])
    return [index_i, index_j]


def coher_indices_motion(image_plane_coords, motion):
    b = motion.shape[0]
    h = motion.shape[2]
    w = motion.shape[3]

    # computing indices to reorder "prev" accordingly to "motion"
    if image_plane_coords is None:
        image_plane_coords = create_image_plane_coords(b, h, w)

    index_i = image_plane_coords[0]
    index_j = image_plane_coords[1]

    if index_i.shape[0] < b:
        index_i = index_i.expand([b, -1, -1])
        index_j = index_j.expand([b, -1, -1])
    elif index_i.shape[0] > b:
        index_i = index_i[0:b, :]
        index_j = index_j[0:b, :]

    ii = (index_i + motion[:, 1, :, :]).to(torch.long)  # b x 1 x h x w
    ii[ii < 0] = 0
    ii[ii >= h] = h - 1

    jj = (index_j + motion[:, 0, :, :]).to(torch.long)  # b x 1 x h x w
    jj[jj < 0] = 0
    jj[jj >= w] = w - 1

    return ii * w + jj


def swap_by_indices(tensor, ii_jj):
    b = tensor.shape[0]
    c = tensor.shape[1]
    h = ii_jj.shape[1]
    w = ii_jj.shape[2]
    prev_original_shape = tensor.shape

    tensor = tensor.view(b, c, h * w)  # b x c x wh

    if b == 1:
        ii_jj = ii_jj.view(-1)  # from b x h x w to flat vector
        tensor = tensor[:, :, ii_jj]  # perhaps indexed_select is faster than torch.gather? (used below)
    else:
        ii_jj = ii_jj.unsqueeze(1).expand(-1, c, -1, -1).view(b, c, h*w)  # b x c x wh
        tensor = torch.gather(tensor, 2, ii_jj)  # b x c x wh

    return tensor.view(prev_original_shape)  # b x c x h x w


def update_average(average, n_average, new_element, n_new_element):
    denominator = n_average + n_new_element
    return average + ((new_element - average) * n_new_element) / denominator, denominator
