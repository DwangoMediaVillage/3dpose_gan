import argparse
import numpy as np
import glob
import os
from progressbar import ProgressBar

from forward_kinematics import _some_variables, fkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/h3.6m')
    args = parser.parse_args()

    paths = list(map(lambda x: glob.glob(os.path.join(x, 'walking_*.txt')),
        glob.glob(os.path.join(args.dir, 'S*'))))
    paths = np.array(paths).flatten().tolist()
    paths.sort()
    pbar = ProgressBar(len(paths))
    for j, path in enumerate(paths):
        data = []
        with open(path) as f:
            for line in f:
                data.append(list(map(float, line.strip().split(','))))
        data = np.array(data)
        parent, offset, rotInd, expmapInd = _some_variables()
        xyz = np.zeros((len(data), 96), dtype=data.dtype)
        for i in range(len(data)):
            xyz[i, :] = fkl(data[i, :], parent, offset, rotInd, expmapInd)
        index = np.array([0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27])
        xyz = xyz.reshape(-1, 32, 3)[:, index]
        np.save(path.replace('.txt', '.npy'), xyz)
        pbar.update(j + 1)
