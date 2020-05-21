from abc import ABC, abstractmethod
import os


class _ForceFieldBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def getAtomTypes(self):
        pass

    @abstractmethod
    def getCharge(self, at):
        pass

    @abstractmethod
    def getMass(self, at):
        pass

    @abstractmethod
    def getLJ(self, at):
        pass

    @abstractmethod
    def getBond(self, at1, at2):
        pass

    @abstractmethod
    def getAngle(self, at1, at2, at3):
        pass

    @abstractmethod
    def getDihedral(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def get14(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def getImproper(self, at1, at2, at3, at4):
        pass


class ForceField:
    def create(mol, prm):
        from torchmd.forcefields.ff_yaml import YamlForcefield
        from torchmd.forcefields.ff_parmed import ParmedForcefield

        parmedext = [".prm", ".prmtop", ".frcmod"]
        yamlext = [".yaml", ".yml"]
        if isinstance(prm, str):
            ext = os.path.splitext(prm)[-1]
            if ext in parmedext:
                return ParmedForcefield(mol, prm)
            elif ext in yamlext:
                return YamlForcefield(mol, prm)
            else:  # Fallback on parmed
                return ParmedForcefield(mol, prm)
        else:  # Fallback on parmed
            return ParmedForcefield(mol, prm)
