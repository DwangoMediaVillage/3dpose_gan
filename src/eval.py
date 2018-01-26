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

def calc_sin(x0, y0, x1, y1):
    xp = cuda.get_array_module(x0)
    l0 = xp.sqrt(xp.power(x0, 2) + xp.power(y0, 2))
    l1 = xp.sqrt(xp.power(x1, 2) + xp.power(y1, 2))
    return (x0 * y1 - x1 * y0) / (l0 * l1)

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
    if opts.nn == 'conv':
        gen = models.net.ConvAE(l_latent=opts.l_latent, l_seq=opts.l_seq,
                                bn=opts.bn,
                                activate_func=getattr(F, opts.act_func),
                                vertical_ksize=opts.vertical_ksize)
    elif opts.nn == 'linear':
        gen = models.net.Linear(
            l_latent=opts.l_latent, l_seq=opts.l_seq,
            bn=opts.bn, activate_func=getattr(F, opts.act_func))
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
            opts.root, action=act_name, length=opts.l_seq,
            train=False, noise_scale=opts.noise_scale)
        test_iter = iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        maes = []
        for batch in test_iter:
            xy, z, scale, noise = dataset.concat_examples(batch, device=args.gpu)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                xy_real = xy + noise
                z_pred = gen(xy_real)

            x_real = xy_real[:, :, :, 0::2]
            # 首から鼻へのzx平面上のベクトル(a0, b0)
            a0 = z_pred.data[:, :, :, 9] - z_pred.data[:, :, :, 8]
            b0 = x_real[:, : ,:, 9] - x_real[:, : ,:, 8]
            # 右肩から左肩へのzx平面上のベクトル(a1, b1)
            a1 = z_pred.data[:, :, :, 14] - z_pred.data[:, :, :, 11]
            b1 = x_real[:, : ,:, 14] - x_real[:, : ,:, 11]
            # 上の2つのベクトルが成す角のsin値．正なら人間として正しい．
            deg_sin = calc_sin(a0, b0, a1, b1)

            # noiseがある場合はnoiseも評価に入れる
            xx = gen.xp.power(noise[:, :, :, 0::2], 2)
            yy = gen.xp.power(noise[:, :, :, 1::2], 2)

            # zを反転しない場合
            zz1 = gen.xp.power(z - z_pred.data, 2)
            m1 = gen.xp.sqrt(xx + yy + zz1).mean(axis=3)[:, 0]

            # zに-1を掛けて反転した場合のLoss
            zz2 = gen.xp.power(z + z_pred.data, 2)
            m2 = gen.xp.sqrt(xx + yy + zz2).mean(axis=3)[:, 0]

            # sin値が負ならzに-1を掛けて反転した場合のLossを使用
            mae = gen.xp.where(deg_sin[:, 0] >= 0, m1, m2)
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
