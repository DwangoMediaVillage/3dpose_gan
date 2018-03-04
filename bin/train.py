#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

from __future__ import print_function
import argparse
import multiprocessing
import pickle
import time

import chainer
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import chainerui.extensions
import chainerui.utils

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projection_gan.pose.posenet import ConvAE, Linear
from projection_gan.pose.pose_dataset import PoseDataset, MPII
from projection_gan.pose.updater import Updater
from projection_gan.pose.evaluator import Evaluator


class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)


def create_result_dir(dir):
    if not os.path.exists('results'):
        os.mkdir('results')
    if dir:
        result_dir = os.path.join('results', dir)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/h3.6m')
    parser.add_argument('--l_latent', type=int, default=64)
    parser.add_argument('--l_seq', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--opt', type=str, default='Adam',
                        choices=['Adam', 'NesterovAG', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--shift_interval', type=int, default=100)
    parser.add_argument('--bn', type=str, default='f', choices=['t', 'f'])
    parser.add_argument('--batch_statistics', type=str, default='f', choices=['t', 'f'])
    parser.add_argument('--train_mode', type=str, default='dcgan',
                        choices=['dcgan', 'wgan', 'supervised'])
    parser.add_argument('--act_func', type=str, default='leaky_relu')
    parser.add_argument('--vertical_ksize', type=int, default=1)
    parser.add_argument('--dcgan_accuracy_cap', type=float, default=1.0,
                        help="Disのaccuracyがこれを超えると更新しない手加減")
    parser.add_argument('--action', '-a', type=str, default='all')
    parser.add_argument('--snapshot_interval', type=int, default=1)
    parser.add_argument('--nn', type=str, default='conv', choices=['linear', 'conv'],
                        help='使用するモデルの選択')
    parser.add_argument('--noise_scale', type=float, default=0,
                        help='xyに加えるガウシアンノイズの大きさ')
    parser.add_argument('--use_heuristic_loss', action="store_true")
    parser.add_argument('--heuristic_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_sh_detection', action="store_true")
    parser.add_argument('--use_mpii', action="store_true")
    args = parser.parse_args()
    args.dir = create_result_dir(args.dir)
    args.bn = args.bn == 't'
    args.batch_statistics = args.batch_statistics == 't'

    # オプションの保存
    with open(os.path.join(args.dir, 'options.pickle'), 'wb') as f:
        pickle.dump(args, f)

    # モデルのセットアップ
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
    if args.nn == 'conv':
        gen = ConvAE(l_latent=args.l_latent, l_seq=args.l_seq, mode='generator',
                     bn=args.bn, activate_func=getattr(F, args.act_func),
                     vertical_ksize=args.vertical_ksize)
        dis = ConvAE(l_latent=1, l_seq=args.l_seq, mode='discriminator',
                     bn=False, activate_func=getattr(F, args.act_func),
                     vertical_ksize=args.vertical_ksize)
    elif args.nn == 'linear':
        gen = Linear(l_latent=args.l_latent, l_seq=args.l_seq, mode='generator',
                     bn=args.bn, activate_func=getattr(F, args.act_func))
        dis = Linear(l_latent=1, l_seq=args.l_seq, mode='discriminator',
                     bn=args.bn, activate_func=getattr(F, args.act_func))
    if args.gpu >= 0:
        gen.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model):
        if args.opt == 'Adam':
            optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
            optimizer.setup(model)
            optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
        elif args.opt == 'NesterovAG':
            optimizer = chainer.optimizers.NesterovAG(lr=args.lr, momentum=0.9)
            optimizer.setup(model)
            optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
        elif args.opt == 'RMSprop':
            optimizer = chainer.optimizers.RMSprop(5e-5)
            optimizer.setup(model)
            optimizer.add_hook(chainer.optimizer.GradientClipping(1))
        else:
            raise NotImplementedError
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
    if args.opt == 'RMSprop':
        opt_dis.add_hook(WeightClipping(0.01))

    # データセットの読み込み
    train = PoseDataset(action=args.action, length=args.l_seq,
                        train=True, use_sh_detection=args.use_sh_detection)
    test = PoseDataset(action=args.action, length=args.l_seq,
                       train=False, use_sh_detection=args.use_sh_detection)
    if args.use_mpii:
        train = MPII(train=True, use_sh_detection=args.use_sh_detection)
        test = MPII(train=False, use_sh_detection=args.use_sh_detection)
    # multiprocessing.set_start_method('spawn')
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.test_batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = Updater(
        mode=args.train_mode,
        batch_statistics=args.batch_statistics,
        models=(gen, dis),
        iterator={'main': train_iter, 'test': test_iter},
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu,
        dcgan_accuracy_cap=args.dcgan_accuracy_cap,
        use_heuristic_loss=args.use_heuristic_loss,
        heuristic_loss_weight=args.heuristic_loss_weight,
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.dir)
    trainer.extend(chainerui.extensions.CommandsExtension())

    log_interval = (1, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'epoch')

    if args.opt == 'NesterovAG':
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, optimizer=opt_gen),
            trigger=(args.shift_interval, 'epoch'))
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, optimizer=opt_dis),
            trigger=(args.shift_interval, 'epoch'))
    trainer.extend(Evaluator(test_iter, {'gen': gen}, device=args.gpu),
                   trigger=log_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt_gen, 'opt_gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt_dis, 'opt_dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/mse',
        'gen/loss', 'gen/loss_heuristic','dis/loss', 'dis/acc', 'dis/acc/real',
        'dis/acc/fake', 'validation/gen/mse',
        'validation/gen/mae1', 'validation/gen/mae2'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    chainerui.utils.save_args(args, args.dir)

    if args.resume:
        import glob
        import re
        # Resume from a snapshot
        epochs = [int(re.search('gen_epoch_(.*).npz', path).group(1)) for path in
                  glob.glob(os.path.join(args.resume, 'gen_epoch_*.npz'))]
        max_epoch = max(epochs)
        chainer.serializers.load_npz(
            os.path.join(args.resume, 'gen_epoch_{}.npz'.format(max_epoch)), gen)
        chainer.serializers.load_npz(
            os.path.join(args.resume, 'dis_epoch_{}.npz'.format(max_epoch)), dis)
        chainer.serializers.load_npz(
            os.path.join(args.resume, 'opt_gen_epoch_{}.npz'.format(max_epoch)), opt_gen)
        chainer.serializers.load_npz(
            os.path.join(args.resume, 'opt_dis_epoch_{}.npz'.format(max_epoch)), opt_dis)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
