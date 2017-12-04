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


def create_fig(out_dir, scope='f_score', model='dec', calc_max=True):
    # load data
    data = json.load(open(os.path.join(out_dir, 'log')))
    train_x = []
    valid_x = []
    train_y = []
    valid_y = []
    for d in data:
        train_x.append(d['epoch'])
        train_y.append(d['{}/{}'.format(model, scope)])
        if '{}/{}'.format(model, scope) in d.keys():
            valid_x.append(d['epoch'])
            valid_y.append(d['{}/{}'.format(model, scope)])

    # show graph
    plt.plot(train_x, train_y, label='train {}'.format(scope))
    plt.plot(valid_x, valid_y, label='valid {}'.format(scope))

    if calc_max:
        plt.title('{0} (max. valid {0}: {1:.4f}({2} epoch))'.format(
            scope, max(valid_y), valid_x[np.array(valid_y).argmax()]))
    else:
        plt.title('{0} (min. valid {0}: {1:.4f}({2} epoch))'.format(
            scope, min(valid_y), valid_x[np.array(valid_y).argmin()]))
    plt.xlabel('epochs')
    plt.ylabel(scope)
    # plt.xlim([0, 60000])
    # plt.ylim([0.016, 0.018])
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(out_dir, 'graph_{}-{}.pdf'.format(model, scope)),
                bbox_inches='tight')
    plt.close()
    print('Saved fig as \'{}\' !!'.format(
        os.path.join(out_dir, 'graph_{}-{}.pdf'.format(model, scope))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str)
    parser.add_argument('--scope', '-s', type=str, default='all')
    parser.add_argument('--model', '-M', type=str, default='dec')
    parser.add_argument('--max', '-m', type=str, default='t',
                        choices=['t', 'f'])
    args = parser.parse_args()
    args.max = args.max == 't'

    if args.scope == 'all':
        create_fig(args.out_dir, 'precision', 'dec', True)
        create_fig(args.out_dir, 'recall', 'dec', True)
        create_fig(args.out_dir, 'f_score', 'dec', True)

        create_fig(args.out_dir, 'loss_rec', 'dec', False)
        create_fig(args.out_dir, 'loss_cls_fake', 'dec', False)
        create_fig(args.out_dir, 'accuracy_cls_fake', 'dec', True)
        create_fig(args.out_dir, 'loss_adv', 'dec', False)
        create_fig(args.out_dir, 'sum_loss', 'dec', False)

        create_fig(args.out_dir, 'loss_cls_real', 'dis', False)
        create_fig(args.out_dir, 'loss_real', 'dis', False)
        create_fig(args.out_dir, 'loss_fake', 'dis', False)
        create_fig(args.out_dir, 'sum_loss', 'dis', False)

        create_fig(args.out_dir, 'accuracy_cls_real', 'dis', True)
        create_fig(args.out_dir, 'accuracy_real', 'dis', True)
        create_fig(args.out_dir, 'accuracy_fake', 'dis', True)
    else:
        create_fig(args.out_dir, args.scope, args.model, args.max)
