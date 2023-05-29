import numpy as np
import torch


class npzMolecule:
    """Class to load npz files in torchmd, essentially a wrapper around the npz file. Necessary information is extracted from the npz file
    and stored in the class, such as the coordinates and the atomic numbers. The class also contains a dictionary to convert atomic numbers to
    atomic symbols. Additional properties such as the box, the charges and the bonds are also stored in the class but are not necessary for the
    simulation."""

    def __init__(self, file: str):
        self.converter = {
            "1": ["H", 1.0080],
            "6": ["C", 12.011],
            "7": ["N", 14.007],
            "8": ["O", 15.999],
            "9": ["F", 18.99840316],
            "15": ["P", 30.97376200],
            "16": ["S", 32.07],
            "17": ["Cl", 35.45],
            "35": ["Br", 79.90],
            "53": ["I", 126.9045],
        }
        self.data = np.load(file)
        self.z = torch.from_numpy(self.data["z"])
        self.coords = torch.from_numpy(self.data["coord"])[:, :, None]
        self.numAtoms = len(self.z)
        self.masses = self.make_masses_npz(self.z)
        self.atomtype = self.get_atom_types(self.z)
        self.embedding = self.z.tolist()

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

    def make_masses_npz(self, z):
        masses = torch.tensor([self.converter[str(el)][1] for el in z.tolist()])
        return masses[:, None]

    def get_atom_types(self, z):
        atomtypes = [self.converter[str(at)][0] for at in z.tolist()]
        return atomtypes
