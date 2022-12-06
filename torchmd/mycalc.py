from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import InputError, ReadError
from ase.calculators.calculator import CalculatorSetupError
from ase import io
import numpy as np
from ase.units import Bohr, Hartree, kcal, mol, Angstrom
import os
import torch


class MyCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        evaluator,
        restart=None,
        ignore_bad_restart=False,
        label="mycalc",
        atoms=None,
        command=None,
        **kwargs
    ):
        Calculator.__init__(
            self,
            restart=restart,
            ignore_bad_restart=ignore_bad_restart,
            label=label,
            atoms=atoms,
            command=command,
            **kwargs
        )
        self.evaluator = evaluator

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=all_changes,
        symmetry="c1",
    ):

        Calculator.calculate(self, atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError(
                "An Atoms object must be provided to " "perform a calculation"
            )
        atoms = self.atoms

        pos = torch.tensor(atoms.positions).double().to("cpu")
        cell = atoms.cell.tolist()
        cell = torch.tensor([cell[0][0], cell[1][1], cell[2][2]]).double().to("cpu")
        energy = self.evaluator.compute(pos, cell)

        # Do the calculations
        if "forces" in properties:
            # energy comes for free
            self.results["energy"] = energy
            # convert to eV/A
            # also note that the gradient is -1 * forces
            self.results["forces"] = self.evaluator.forces.cpu().numpy()
        elif "energy" in properties:
            # convert to eV
            self.results["energy"] = energy
