import glob
import csv
import json
import os
import time
import argparse
import torch
import yaml
import sys
import numpy as np


class LogWriter(object):
    #kind of inspired form openai.baselines.bench.monitor
    #We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header=''):
        self.keys = tuple(keys)+('t',)
        assert path is not None
        self._clean_log_dir(path)
        filename = os.path.join(path, 'monitor.csv')
        print("Writing logs to ",filename)

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo['t'] = t
            self.logger.writerow(epinfo)
            self.f.flush()

    def _clean_log_dir(self,log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            files = glob.glob(os.path.join(log_dir, '*.csv'))
            for f in files:
                os.remove(f)


class LoadFromFile(argparse.Action):
#parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split('='), namespace)


def save_argparse(args,filename):
    with open(filename, 'w') as f:
        for k,v in args.__dict__.items():
            f.write(f'{k}={v}\n')



