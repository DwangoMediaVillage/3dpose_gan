#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class ConvAE(chainer.Chain):
    def __init__(self, l_latent=64, l_seq=32, mode='generator', bn=False,
                 activate_func=F.leaky_relu, vertical_ksize=1):

        if l_seq % 32 != 0:
            raise ValueError('\'l_seq\' must be divisible by 32.')
        if not mode in ['discriminator', 'generator']:
            raise ValueError('\'mode\' must be \'discriminator\' or \'generator\'.')

        super(ConvAE, self).__init__()
        self.bn = bn
        self.mode = mode
        self.activate_func = activate_func
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
            self.conv2 = L.Convolution2D(32, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
            self.conv3 = L.Convolution2D(64, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
            self.conv4 = L.Convolution2D(64, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
            self.conv5 = L.Convolution2D(64, 128, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
            self.enc_l = L.Linear(128 * 34 * l_seq // 32, l_latent, initialW=w)

            if self.mode == 'generator':
                self.dec_l = L.Linear(l_latent, 128 * 17 * l_seq // 32, initialW=w)
                self.deconv1 = L.Deconvolution2D(128, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
                self.deconv2 = L.Deconvolution2D(64, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
                self.deconv3 = L.Deconvolution2D(64, 64, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
                self.deconv4 = L.Deconvolution2D(64, 32, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)
                self.deconv5 = L.Deconvolution2D(32, 1, ksize=(4, vertical_ksize), stride=(2, 1), pad=(1, (vertical_ksize-1)//2), initialW=w)

            if self.bn:
                self.enc_bn1 = L.BatchNormalization(32)
                self.enc_bn2 = L.BatchNormalization(64)
                self.enc_bn3 = L.BatchNormalization(64)
                self.enc_bn4 = L.BatchNormalization(64)
                self.enc_bn5 = L.BatchNormalization(128)

                if self.mode == 'generator':
                    self.dec_bn1 = L.BatchNormalization(128)
                    self.dec_bn2 = L.BatchNormalization(64)
                    self.dec_bn3 = L.BatchNormalization(64)
                    self.dec_bn4 = L.BatchNormalization(64)
                    self.dec_bn5 = L.BatchNormalization(32)

    def __call__(self, x):
        h = self.encode(x)
        if self.mode == 'generator':
            h = self.decode(h)
        return h

    def encode(self, x):
        for i in range(1, 6):
            x = self['conv{}'.format(i)](x)
            if self.bn:
                x = self['enc_bn{}'.format(i)](x)
            x = self.activate_func(x)
        self.b, self.c, self.n, self.p = x.shape
        h = F.reshape(x, (self.b, self.c * self.n * self.p))
        h = self.enc_l(h)
        return h

    def decode(self, h):
        h = self.dec_l(h)
        h = F.reshape(h, (self.b, self.c, self.n, self.p // 2))
        for i in range(1, 6):
            if self.bn:
                h = self['dec_bn{}'.format(i)](h)
            h = self.activate_func(h)
            h = self['deconv{}'.format(i)](h)
        return h


class Linear(chainer.Chain):
    def __init__(self, l_latent=64, l_seq=1, unit=1024, mode='generator',
                 bn=False, activate_func=F.relu):
        super(Linear, self).__init__()
        n_out = l_seq * 17 if mode == 'generator' else 1
        print('MODEL: {}, N_OUT: {}, N_UNIT: {}'.format(mode, n_out, unit))
        self.l_seq = l_seq
        self.mode = mode
        self.bn = bn
        self.activate_func = activate_func
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l1 = L.Linear(l_seq * 34, unit, initialW=w)
            self.l2 = L.Linear(unit, unit, initialW=w)
            self.l3 = L.Linear(unit, unit, initialW=w)
            self.l4 = L.Linear(unit, unit, initialW=w)
            self.l5 = L.Linear(unit, unit, initialW=w)
            self.l6 = L.Linear(unit, unit, initialW=w)
            self.l7 = L.Linear(unit, unit, initialW=w)
            self.l8 = L.Linear(unit, n_out, initialW=w)

            if self.bn:
                self.bn1 = L.BatchNormalization(unit)
                self.bn2 = L.BatchNormalization(unit)
                self.bn3 = L.BatchNormalization(unit)
                # self.bn4 = L.BatchNormalization(unit)
                # self.bn5 = L.BatchNormalization(unit)
                # self.bn6 = L.BatchNormalization(unit)
                # self.bn7 = L.BatchNormalization(unit)

    def __call__(self, x):
        x = F.reshape(x, (len(x), -1))
        if self.bn:
            h1 = self.activate_func(self.bn1(self.l1(x)))
            h2 = self.activate_func(self.bn2(self.l2(h1)))
            h = self.activate_func(self.bn3(self.l3(h2)) + h1)
            # h4 = self.activate_func(self.bn4(self.l4(h)))
            # h5 = self.activate_func(self.bn5(self.l5(h4)) + h)
            # h6 = self.activate_func(self.bn6(self.l6(h5)))
            # h = self.activate_func(self.bn7(self.l7(h6)) + h5)
            h8 = self.l8(h)
        else:
            h1 = self.activate_func(self.l1(x))
            h2 = self.activate_func(self.l2(h1))
            h = self.activate_func(self.l3(h2) + h1)
            # h4 = self.activate_func(self.l4(h3))
            # h5 = self.activate_func(self.l5(h4) + h3)
            # h6 = self.activate_func(self.l6(h5))
            # h = self.activate_func(self.l7(h6) + h5)
            h8 = self.l8(h)
        if self.mode == 'generator':
            h8 = F.reshape(h8, (len(h8), 1, self.l_seq, 17))
        return h8
