import csv
import json
import os
import time
import argparse
import yaml


class LogWriter(object):
    # kind of inspired form openai.baselines.bench.monitor
    # We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header="", name="monitor.csv"):
        self.keys = tuple(keys) + ("t",)
        assert path is not None

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, name)
        if os.path.exists(filename):
            os.remove(filename)

        print("Writing logs to ", filename)

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = "# {} \n".format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo["t"] = t
            self.logger.writerow(epinfo)
            self.f.flush()


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
                return

        with values as f:
            input = f.read()
            input = input.rstrip()
            for lines in input.split("\n"):
                k, v = lines.split("=")
                typ = type(namespace.__dict__[k])
                v = typ(v) if typ is not None else v
                namespace.__dict__[k] = v


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [
                exclude,
            ]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        with open(filename, "w") as fout:
            yaml.dump(args, fout)
    else:
        with open(filename, "w") as f:
            for k, v in args.__dict__.items():
                if k is exclude:
                    continue
                f.write(f"{k}={v}\n")

def compute_max_interatomic_dist(currpos):
    #currpos.shape = (N_replicas, Natomos, 3)
    N_replicas=currpos.shape[0]
    N_atoms=currpos.shape[1]
    max_i_d=np.zeros([N_replicas])
    distance_matrix=np.zeros([N_replicas,N_atoms,N_atoms])
    for rep in range(N_replicas):
        for ind1 in range(Nat):
            for ind2 in range(ind1+1,Nat):
                distance_matrix[rep,ind1,ind2]=np.linalg.norm(currpos[rep,ind1,:]-currpos[rep,ind2,:])
        max_i_d[rep]=np.max(distance_matrix[rep,:,:])

    return max_i_d
    