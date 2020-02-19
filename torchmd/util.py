import torch


def wrap_dist(dist, box):
    if torch.all(box == 0):
        return dist
    return dist - box[None, :] * torch.round(dist / box[None, :])

def calculateDistances(atom_pos, atom_idx1, atom_idx2, box):
    direction_vec = wrap_dist(atom_pos[atom_idx1, :] - atom_pos[atom_idx2, :], box)
    dist = torch.sqrt(torch.sum(direction_vec * direction_vec, dim=1))
    direction_unitvec = direction_vec / dist[:, None]
    return dist, direction_unitvec

