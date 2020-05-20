from abc import ABC, abstractmethod


class ForceField(ABC):
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
