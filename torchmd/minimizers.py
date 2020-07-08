import torch
import numpy as np

# def minimize_pytorch(system, forces, fmax=0.05, steps=1e9):
#     opt = torch.optim.SGD(system.pos)

#     curr_fmax = torch.float("Inf")
#     curr_step = 0
#     while curr_fmax > fmax:
#         Epot = forces.compute(system.pos, system.box, system.forces)

#         opt.zero_grad()
#         Epot.backward()

#         curr_force = -system.pos.grad
#         curr_fmax = torch.max(torch.norm(curr_force, dim=2), dim=1)

#         opt.step()

#         curr_step += 1
#         if curr_step > steps:
#             break


def minimize_bfgs(system, forces, fmax=0.5, steps=1000):
    from scipy.optimize import minimize

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
    x0 = system.pos.detach().cpu().numpy()[0].astype(np.float64)
    res = minimize(
        evalfunc,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": fmax, "maxiter": steps, "disp": False},
        args=({"Nfeval": 0},),
    )

    system.pos = torch.tensor(
        res.x.reshape(1, -1, 3), dtype=system.pos.dtype, device=system.pos.device
    )
