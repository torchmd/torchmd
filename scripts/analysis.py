
import pandas as pd
import numpy as np
from moleculekit.molecule import Molecule

structure='./tests/argon/argon2.pdb'
df = pd.read_csv('mytest/monitor.csv')
df.plot(x='ns',y=['etot'])

mol=Molecule(structure)
coords = np.load('mytest/output.npy')
mol.coords=coords
mol.reps.add('all','vdw')
mol.view()