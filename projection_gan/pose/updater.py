#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import Variable

import numpy as np


class Updater(chainer.training.StandardUpdater):
    def __init__(self, dcgan_accuracy_cap, *args, use_heuristic_loss, heuristic_loss_weight: float, **kwargs):
        self.heuristic_loss_weight = heuristic_loss_weight
        self.use_heuristic_loss = use_heuristic_loss
        self.dcgan_accuracy_cap = dcgan_accuracy_cap
        self.mode = kwargs.pop('mode')
        self.batch_statistics = kwargs.pop('batch_statistics')
        if not self.mode in ['dcgan', 'wgan', 'supervised']:
            raise ValueError
        self.gen, self.dis = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    @staticmethod
    def calculate_rotation(xy_real: chainer.Variable, z_pred: chainer.Variable) -> chainer.Variable:
        xy_split = chainer.functions.split_axis(xy_real, xy_real.data.shape[3], axis=3)
        z_split = chainer.functions.split_axis(z_pred, z_pred.data.shape[3], axis=3)

        # 首から鼻へのzx平面上のベクトル(a0, b0)
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = chainer.functions.sqrt(a0 * a0 + b0 * b0)
        # 右肩から左肩へのzx平面上のベクトル(a1, b1)
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = chainer.functions.sqrt(a1 * a1 + b1 * b1)
        # 肩とのなす角
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def calculate_heuristic_loss(xy_real: chainer.Variable, z_pred: chainer.Variable) -> chainer.Variable:
        return chainer.functions.average(
            chainer.functions.relu(- Updater.calculate_rotation(xy_real, z_pred))
        )

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = self.gen, self.dis

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        xy, xyz, scale = chainer.dataset.concat_examples(batch, self.device)

        xy_real = Variable(xy)
        z_pred = gen(xy_real)

        # Random rotation.
        theta = np.random.uniform(0, 2 * np.pi, len(xy)).astype(np.float32)
        cos_theta = np.broadcast_to(np.cos(theta), z_pred.shape[::-1])
        cos_theta = Variable(self.gen.xp.array(cos_theta.transpose(3, 2, 1, 0)))
        sin_theta = np.broadcast_to(np.sin(theta), z_pred.shape[::-1])
        sin_theta = Variable(self.gen.xp.array(sin_theta.transpose(3, 2, 1, 0)))

        # 2D Projection.
        x = xy_real[:, :, :, 0::2]
        y = xy_real[:, :, :, 1::2]
        xx = x * cos_theta + z_pred * sin_theta
        xx = xx[:, :, :, :, None]
        yy = y[:, :, :, :, None]
        xy_fake = F.concat((xx, yy), axis=4)
        xy_fake = F.reshape(xy_fake, (*y.shape[:3], -1))

        if self.batch_statistics:
            xy_real = concat_stat(xy_real)
            xy_fake = concat_stat(xy_fake)

        y_real = dis(xy_real)
        y_fake = dis(xy_fake)
        mse = F.mean_squared_error(z_pred, xyz[:, :, :, 2::3])

        if self.mode == 'supervised':
            gen.cleargrads()
            mse.backward()
            gen_optimizer.update()
            chainer.report({'mse': mse}, gen)

        elif self.mode == 'dcgan':
            acc_dis_fake = F.binary_accuracy(y_fake, dis.xp.zeros(y_fake.data.shape, dtype=int))
            acc_dis_real = F.binary_accuracy(y_real, dis.xp.ones(y_real.data.shape, dtype=int))
            acc_dis = (acc_dis_fake + acc_dis_real) / 2

            loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
            if self.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(xy_real=xy_real, z_pred=z_pred)
                loss_gen += loss_heuristic * self.heuristic_loss_weight
                chainer.report({'loss_heuristic': loss_heuristic}, gen)
            gen.cleargrads()
            if acc_dis.data >= (1 - self.dcgan_accuracy_cap):
                loss_gen.backward()
                gen_optimizer.update()
            xy_fake.unchain_backward()

            loss_dis = F.sum(F.softplus(-y_real)) / batchsize
            loss_dis += F.sum(F.softplus(y_fake)) / batchsize
            dis.cleargrads()
            if acc_dis.data <= self.dcgan_accuracy_cap:
                loss_dis.backward()
                dis_optimizer.update()

            chainer.report({'loss': loss_gen, 'mse': mse}, gen)
            chainer.report({'loss': loss_dis, 'acc': acc_dis, 'acc/fake': acc_dis_fake, 'acc/real': acc_dis_real}, dis)

        elif self.mode == 'wgan':
            y_real = F.sum(y_real) / batchsize
            y_fake = F.sum(y_fake) / batchsize

            wasserstein_distance = y_real - y_fake
            loss_dis = -wasserstein_distance
            loss_gen = -y_fake
            if self.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(xy_real=xy_real, z_pred=z_pred)
                loss_gen += loss_heuristic * self.heuristic_loss_weight
                chainer.report({'loss_heuristic': loss_heuristic}, gen)

            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

            if self.iteration < 2500 and self.iteration % 100 == 0:
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()

            if self.iteration > 2500 and self.iteration % 5 == 0:
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()

            chainer.report({'loss': loss_gen, 'mse': mse}, gen)
            chainer.report({'loss': loss_dis}, dis)

        else:
            raise NotImplementedError


def concat_stat(x):
    mean = F.mean(x, axis=0)
    mean = F.concat([mean[None]] * x.shape[0], axis=0)
    variance = F.broadcast_to(F.mean((x - mean) * (x - mean)), x.shape)
    return F.concat((x, variance), axis=1)
