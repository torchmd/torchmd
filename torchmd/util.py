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

def wrap_coords(pos, box, wrapidx, groups, nongrouped):
    nmol = len(groups)

    if wrapidx is not None:
        # Get COM of wrapping center group
        com = torch.sum(pos[wrapidx], dim=0) / len(wrapidx)
        # Subtract COM from all atoms so that the center mol is at [box/2, box/2, box/2]
        pos = (pos - com) + (box / 2)

    if nmol != 0:
        # Work out the COMs and offsets of every group and move group to [0, box] range
        for i, group in enumerate(groups):
            tmp_com = torch.sum(pos[group], dim=0) / len(group)
            offset = torch.floor(tmp_com / box) * box
            pos[group] -= offset

    # Move non-grouped atoms
    offset = torch.floor(pos[nongrouped] / box) * box
    pos[nongrouped] -= offset
