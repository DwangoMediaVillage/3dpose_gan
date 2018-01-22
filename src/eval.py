#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Yasunori Kudo

import chainer
from chainer import cuda
from chainer import dataset
from chainer import iterators
from chainer import serializers

import chainer.functions as F

import argparse
import os
import pickle
from progressbar import ProgressBar
import sys

sys.path.append(os.getcwd())
import src.dataset
import models.net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Generatorの重みファイルへのパス')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=200)
    args = parser.parse_args()

    # 学習時のオプションの読み込み
    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.pickle'), 'rb') as f:
        opts = pickle.load(f)

    # モデルの定義
    gen = models.net.ConvAE(l_latent=opts.l_latent, l_seq=opts.l_seq,
                            bn=opts.bn,
                            activate_func=getattr(F, opts.act_func),
                            vertical_ksize=opts.vertical_ksize)
    serializers.load_npz(args.model_path, gen)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()

    # 行動クラスの読み込み
    if opts.action == 'all':
        with open('data/actions.txt') as f:
            actions = f.read().split('\n')[:-1]
    else:
        actions = [opts.action]

    # 各行動クラスに対して平均エラー(mm)を算出
    errors = []
    for act_name in actions:
        test = src.dataset.PoseDataset(
            opts.root, action=act_name, length=opts.l_seq, train=False)
        test_iter = iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        maes = []
        for batch in test_iter:
            xy, z, scale = dataset.concat_examples(batch, device=args.gpu)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                z_pred = gen(xy)
            mae = gen.xp.abs(z - z_pred.data).mean(axis=3)[:, 0]
            mae *= scale
            mae = gen.xp.mean(mae)
            maes.append(mae * len(batch))
        test_iter.finalize()
        print(act_name, sum(maes) / len(test))
        errors.append(sum(maes) / len(test))
    print('-' * 20)
    print('average', sum(errors) / len(errors))

    # csvとして保存
    with open(args.model_path.replace('.npz', '.csv'), 'w') as f:
        for act_name, error in zip(actions, errors):
            f.write('{},{}\n'.format(act_name, error))
        f.write('{},{}\n'.format('average', sum(errors) / len(errors)))
