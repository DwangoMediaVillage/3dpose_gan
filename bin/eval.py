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
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Generatorの重みファイルへのパス')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=200)
    parser.add_argument('--allow_inversion', action="store_true",
                         help='評価時にzの反転を許可するかどうか')
    args = parser.parse_args()

    # 学習時のオプションの読み込み
    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.pickle'), 'rb') as f:
        opts = pickle.load(f)

    # モデルの定義
    if opts.nn == 'conv':
        gen = projection_gan.pose.posenet.ConvAE(l_latent=opts.l_latent, l_seq=opts.l_seq,
                                                 bn=opts.bn,
                                                 activate_func=getattr(F, opts.act_func),
                                                 vertical_ksize=opts.vertical_ksize)
    elif opts.nn == 'linear':
        gen = projection_gan.pose.posenet.Linear(
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
        test = projection_gan.pose.dataset.pose_dataset.PoseDataset(
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

            deg_sin = projection_gan.pose.updater.Updater.calculate_rotation(
                chainer.Variable(xy_real), z_pred).data[:, :, :, 0]

            # noiseがある場合はnoiseも評価に入れる
            xx = gen.xp.power(noise[:, :, :, 0::2], 2)
            yy = gen.xp.power(noise[:, :, :, 1::2], 2)

            # zを反転しない場合
            zz1 = gen.xp.power(z - z_pred.data, 2)
            m1 = gen.xp.sqrt(xx + yy + zz1).mean(axis=3)[:, 0]

            if args.allow_inversion:
                # zに-1を掛けて反転した場合のLoss
                zz2 = gen.xp.power(z + z_pred.data, 2)
                m2 = gen.xp.sqrt(xx + yy + zz2).mean(axis=3)[:, 0]

                # sin値が負ならzに-1を掛けて反転した場合のLossを使用
                mae = gen.xp.where(deg_sin[:, 0] >= 0, m1, m2)
            else:
                mae = m1

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
