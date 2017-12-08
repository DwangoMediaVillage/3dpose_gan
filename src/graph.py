#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def create_fig(out_dir, targets, graph_name, calc_max=True, ylim=None):
    # load data
    data = json.load(open(os.path.join(out_dir, 'log')))
    title = ''
    for target in targets:
        info_x, info_y = [], []
        for d in data:
            info_x.append(d['epoch'])
            info_y.append(d[target])
        plt.plot(info_x, info_y, label=target)
        if calc_max:
            title += 'Max {0}: {1:.4f}({2} epoch))\n'.format(
                target, max(info_y), info_x[np.array(info_y).argmax()])
        else:
            title += 'Min {0}: {1:.4f}({2} epoch))\n'.format(
                target, min(info_y), info_x[np.array(info_y).argmin()])
    plt.title(title.strip())
    plt.xlabel('epochs')
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc='upper right')
    if not os.path.exists(os.path.join(out_dir, 'graph')):
        os.mkdir(os.path.join(out_dir, 'graph'))
    plt.savefig(os.path.join(out_dir, 'graph', graph_name),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str)
    args = parser.parse_args()

    targets = ['gen/loss', 'dis/loss']
    create_fig(args.out_dir, targets, 'loss.pdf', calc_max=False)
    targets = ['gen/mse', 'validation/gen/mse']
    create_fig(args.out_dir, targets, 'mse.pdf', calc_max=False, ylim=[0, 0.2])
