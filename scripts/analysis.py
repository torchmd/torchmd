import pandas as pd
import numpy as np
from moleculekit.molecule import Molecule
import sys
import os
import matplotlib.pylab as plt
import yaml

plt.ion()

conf = yaml.load(open(sys.argv[1], "r"), Loader=yaml.FullLoader)

df = pd.read_csv(os.path.join(conf["log_dir"], "monitor.csv"))
df.plot(x="ns", y="etot")
df.plot(x="ns", y="ekin")
df.plot(x="ns", y="epot")

mol = Molecule(conf["structure"])
coords = np.load(os.path.join(conf["log_dir"], "output.npy"))
mol.coords = coords
mol.reps.add("all", "vdw")
mol.view()

input()
