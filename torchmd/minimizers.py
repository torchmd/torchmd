import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def minimize_bfgs(system, forces, fmax=0.5, steps=1000):
    from scipy.optimize import minimize

    if steps == 0:
        return

    if system.pos.shape[0] != 1:
        raise RuntimeError(
            "System minimization currently doesn't support replicas. Talk with Stefan to implement it."
        )

    def evalfunc(coords, info):
        coords = coords.reshape(1, -1, 3)
        coords = torch.tensor(coords).type_as(system.pos)
        Epot = forces.compute(coords, system.box, system.forces)[0]
        grad = -system.forces.detach().cpu().numpy().astype(np.float64)[0]
        # display information
        if info["Nfeval"] % 1 == 0:
            print(
                "{0:4d}   {1: 3.6f}   {2: 3.6f}".format(
                    info["Nfeval"], Epot, np.max(np.linalg.norm(grad, axis=1))
                )
            )
        info["Nfeval"] += 1
        return Epot, grad.reshape(-1)

    print("{0:4s} {1:9s}       {2:9s}".format("Iter", " Epot", " fmax"))
    x0 = system.pos.detach().cpu().numpy()[0].astype(np.float64).flatten()

    res = minimize(
        evalfunc,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": fmax, "maxiter": steps, "disp": False},
        args=({"Nfeval": 0},),
    )

    system.pos = torch.tensor(
        res.x.reshape(1, -1, 3),
        dtype=system.pos.dtype,
        device=system.pos.device,
        requires_grad=system.pos.requires_grad,
    )


def minimize_pytorch_bfgs(
    system, calculator, steps=10, max_iter=20, tolerance_change=1e-9
):
    if steps == 0:
        return

    # Work with [N, 3] shape like optimize_geometries
    pos_flat = system.pos.view(-1, 3).clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [pos_flat], max_iter=max_iter, tolerance_change=tolerance_change
    )

    energies = []

    def closure(step):
        opt.zero_grad()

        # Reshape for calculator which expects [nreplicas, natoms, 3]
        pos_reshaped = pos_flat.view(system.nreplicas, system.natoms, 3)
        Epot = calculator.compute(
            pos_reshaped, system.box, system.forces, toNumpy=False
        )
        Epot = torch.stack(Epot)
        energies.append(Epot.detach().cpu().numpy())

        Epot = torch.sum(Epot)  # Sum over all replicas
        Epot.backward()  # Compute gradients for the minimizer

        maxgrad = np.max(np.linalg.norm(pos_flat.grad.detach().cpu().numpy(), axis=1))
        print("{0:4d}   {1: 3.6f}   {2: 3.6f}".format(step[0], float(Epot), maxgrad))
        step[0] += 1
        return Epot

    print("{0:4s} {1:9s}       {2:9s}".format("Iter", " Epot", " fmax"))
    step = [0]
    for i in range(steps):
        opt.step(lambda: closure(step))

    # Update system with reshaped positions
    system.pos[:] = pos_flat.detach().view(1, -1, 3).requires_grad_(False)

    energies = np.concatenate(energies, axis=2)
    return energies


def _get_energy_forces(forces, system, pos, getForces=True):
    Epot = forces.compute(pos, system.box, system.forces)[0]
    if getForces:
        frc = system.forces.detach()[0]
        return Epot, frc
    else:
        return Epot


def _bracket_and_golden_section_search(forces, system, initpos, search_dir, u):
    """
    Bracket and golden section search algorithm.

    Parameters
    ----------
    forces: torch.Tensor
        Forces on the atoms
    system: System
        System object
    initpos: torch.Tensor
        Initial position
    search_dir: torch.Tensor
        Search direction
    u: float
        Should be initialized to be potential for pos, returns potential for min energy pos
    """
    tau = 0.618033988749895  # tau=(sqrt(5)-1)/2,  solution to  tau^2 = 1-tau
    dis = 1.0  # largest displacement along search direction
    tol = 1.0e-2  # tolerance for convergence of search interval
    u_amin = u

    # use s and dis2 to determine amax search factor
    smax2 = torch.max(torch.sum(search_dir**2, dim=1))
    smax = torch.sqrt(smax2)

    amax = dis / smax
    amin = 0.0
    delta = amax - amin

    a1 = amin + (1 - tau) * delta
    a2 = amin + tau * delta

    # interval is considered trivially bracketed if small enough
    is_bracket = (delta * smax) <= tol

    # find potential for amax
    u_amax = _get_energy_forces(forces, system, initpos + amax * search_dir, False)

    # find potential for a1
    u_a1 = _get_energy_forces(forces, system, initpos + a1 * search_dir, False)

    # find potential for a2
    u_a2, frc = _get_energy_forces(forces, system, initpos + a2 * search_dir, True)

    # save most recent computation
    u = u_a2

    while not is_bracket:
        if u_a1 >= u_amin:
            # shrink bracketing interval to [amin,a1]
            # compute new u_a1, u_a2
            amax = a1
            u_amax = u_a1

            delta = amax - amin
            a1 = amin + (1 - tau) * delta
            a2 = amin + tau * delta

            # find potential for a1
            pos = initpos + a1 * search_dir
            u_a1 = _get_energy_forces(forces, system, pos, False)

            # find potential for a2
            pos = initpos + a2 * search_dir
            u_a2, frc = _get_energy_forces(forces, system, pos, True)

            # update is_bracket since interval has shrunk
            is_bracket = delta * smax <= tol

            # save most recent computation
            u = u_a2
        elif u_a2 >= u_amin:
            # shrink bracketing interval to [amin,a2]
            # compute new u_a1
            amax = a2
            u_amax = u_a2
            a2 = a1
            u_a2 = u_a1

            delta = amax - amin
            a1 = amin + (1 - tau) * delta

            # find potential for a1
            pos = initpos + a1 * search_dir
            u_a1, frc = _get_energy_forces(forces, system, pos, True)

            # update is_bracket since interval has shrunk
            is_bracket = delta * smax <= tol

            # save most recent computation
            u = u_a1
        elif u_amax < u_a1 and u_amax < u_a2:
            # shift bracketing interval to [a2,a2+delta]
            # compute new u_a2, u_amax
            amin = a2
            u_amin = u_a2
            a1 = amax
            u_a1 = u_amax

            amax = amin + delta
            a2 = amin + tau * delta

            # find potential for amax
            pos = initpos + amax * search_dir
            u_amax = _get_energy_forces(forces, system, pos, False)

            # find potential for a2
            pos = initpos + a2 * search_dir
            u_a2, frc = _get_energy_forces(forces, system, pos, True)
        else:
            # now we consider bracketed interval unimodal
            # continue with golden section search
            is_bracket = True

    # golden section search
    while delta * smax > tol:
        if u_a1 > u_a2:
            amin = a1
            u_amin = u_a1
            delta = amax - amin

            a1 = a2
            u_a1 = u_a2

            a2 = amin + tau * delta

            # find potential for a2
            pos = initpos + a2 * search_dir
            u_a2, frc = _get_energy_forces(forces, system, pos, True)

            # save most recent computation
            u = u_a2
        else:
            amax = a2
            u_amax = u_a2
            delta = amax - amin

            a2 = a1
            u_a2 = u_a1

            a1 = amin + (1 - tau) * delta

            # find potential for a1
            pos = initpos + a1 * search_dir
            u_a1, frc = _get_energy_forces(forces, system, pos, True)

            # save most recent computation
            u = u_a1

    assert frc is not None
    assert pos is not None

    return pos, frc, u


def minimize_cg(system, forces, steps=1000, start_step: int = 0, threshold=None):
    pos = system.pos.detach().requires_grad_(True)

    # compute initial force
    u, frc = _get_energy_forces(forces, system, pos, getForces=True)

    # use force to set initial search direction
    search_dir = frc.clone().detach()

    # find f dot f
    fdf = torch.sum(frc**2)

    # conjugate gradient loop
    for step in range(start_step, steps):

        # retain initial position
        initpos = pos.clone().detach()

        # find minimum along search direction
        pos, frc, u = _bracket_and_golden_section_search(
            forces, system, initpos, search_dir, u
        )

        old_fdf = fdf

        # find f dot f
        fdf = torch.sum(frc**2)

        # determine new search direction
        beta = fdf / old_fdf
        maxforce = torch.max(torch.abs(frc))

        search_dir = frc + beta * search_dir

        energy, frc = _get_energy_forces(forces, system, pos, True)

        # print results
        maxforce = torch.max(torch.abs(frc))
        logger.info(f"{step:12d} {energy:14.4f} {maxforce:16.4f}")

        # terminate
        if threshold is not None and maxforce < threshold:
            return step

    return steps - 1
