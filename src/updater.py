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

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = self.gen, self.dis

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        xy, z = chainer.dataset.concat_examples(batch, self.device)

        xy = Variable(xy)
        z_pred = gen(xy)

        # Random rotation.
        theta = np.random.uniform(0, 2 * np.pi, len(xy)).astype(np.float32)
        cos_theta = np.broadcast_to(np.cos(theta), z_pred.shape[::-1])
        cos_theta = Variable(cuda.to_gpu(cos_theta.transpose(3, 2, 1, 0)))
        sin_theta = np.broadcast_to(np.sin(theta), z_pred.shape[::-1])
        sin_theta = Variable(cuda.to_gpu(sin_theta.transpose(3, 2, 1, 0)))

        # 2D Projection.
        x = xy[:, :, :, :17]
        y = xy[:, :, :, 17:]
        xy_fake = F.concat((x * cos_theta + z_pred * sin_theta, y), axis=3)

        y_real = dis(xy)
        y_fake = dis(xy_fake)
        mse = F.mean_squared_error(z_pred, z)

        loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        xy_fake.unchain_backward()

        loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.report({'loss': loss_gen, 'mse': mse}, gen)
        chainer.report({'loss': loss_dis}, dis)
