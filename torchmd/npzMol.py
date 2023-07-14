import numpy as np
import torch
from moleculekit.periodictable import periodictable_by_number


class npzMolecule:
    def __init__(self, file: str):
        """Class to load npz files in torchmd, essentially a wrapper around the npz file. Necessary information is extracted from the npz file
        and stored in the class, such as the coordinates and the atomic numbers. The class also contains a dictionary to get atom masses from
        atomic numbers. Additional properties such as the box, the charges and the bonds are also stored in the class but are not necessary for
        the simulation."""

        self.data = np.load(file)
        self.z = torch.from_numpy(self.data["z"])
        self.coords = torch.from_numpy(self.data["coord"])[:, :, None]
        self.numAtoms = len(self.z)
        self.embedding = self.z.tolist()

        self.masses = torch.tensor(
            [periodictable_by_number[int(el)].mass for el in self.z.tolist()]
        ).to(torch.float32)[:, None]

        self.element = np.array(
            [periodictable_by_number[int(el)].symbol for el in self.z.tolist()]
        ).astype("object")
        self.atomtype = self.element.copy()

        if "charges" in self.data.files:
            self.charge = torch.from_numpy(self.data["charges"])
        else:
            self.charge = np.zeros_like(self.z)

        if "bonds" in self.data.files:
            self.bonds = torch.from_numpy(self.data["bonds"])
        else:
            self.bonds = []

        if "box" in self.data.files:
            self.box = self.data["box"]
        else:
            self.box = np.zeros((3, 1))
