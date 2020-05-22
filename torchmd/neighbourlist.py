import torch


def discretize_box(box, subcell_size):
    xbins = torch.arange(0, box[0, 0] + subcell_size, subcell_size)
    ybins = torch.arange(0, box[1, 1] + subcell_size, subcell_size)
    zbins = torch.arange(0, box[2, 2] + subcell_size, subcell_size)
    nxbins = len(xbins) - 1
    nybins = len(ybins) - 1
    nzbins = len(zbins) - 1

    r = torch.tensor([-1, 0, 1])
    neighbour_mask = torch.cartesian_prod(r, r, r)

    cellidx = torch.cartesian_prod(
        torch.arange(nxbins), torch.arange(nybins), torch.arange(nzbins)
    )
    cellneighbours = cellidx.unsqueeze(2) + neighbour_mask.T.unsqueeze(0).repeat(
        cellidx.shape[0], 1, 1
    )

    # Can probably be done easier as we only need to handle -1 and max cases, not general -2, max+1 etc
    nbins = torch.tensor([nxbins, nybins, nzbins])[None, :, None].repeat(
        cellidx.shape[0], 1, 27
    )
    negvals = cellneighbours < 0
    cellneighbours[negvals] += nbins[negvals]
    largevals = cellneighbours > (nbins - 1)
    cellneighbours[largevals] -= nbins[largevals]

    return xbins, ybins, zbins, cellneighbours


def neighbour_list(pos, box, subcell_size):
    nsystems = coordinates.shape[0]

    for s in range(nsystems):
        spos = pos[s]
        sbox = box[s]

        xbins, ybins, zbins = discretize_box(sbox, subcell_size)

        xidx = torch.bucketize(spos[:, 0], xbins, out_int32=True)
        yidx = torch.bucketize(spos[:, 1], ybins, out_int32=True)
        zidx = torch.bucketize(spos[:, 2], zbins, out_int32=True)

        binidx = torch.stack((xidx, yidx, zidx)).T
