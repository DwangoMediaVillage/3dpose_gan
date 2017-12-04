#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class ConvAE(chainer.Chain):

    def __init__(self, l_latent=64, l_seq=32, mode='generator', bn=False,
                 activate_func=F.leaky_relu):

        if l_seq % 32 != 0:
            raise ValueError('\'l_seq\' must be divisible by 32.')
        if not mode in ['discriminator', 'generator']:
            raise ValueError('\'mode\' must be \'discriminator\' or \'generator\'.')

        super(ConvAE, self).__init__()
        self.bn = bn
        self.mode = mode
        self.activate_func = activate_func
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
            self.conv2 = L.Convolution2D(32, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
            self.conv3 = L.Convolution2D(64, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
            self.conv4 = L.Convolution2D(64, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
            self.conv5 = L.Convolution2D(64, 128, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
            self.enc_l = L.Linear(128 * 34 * l_seq // 32, l_latent)

            if self.mode == 'generator':
                self.dec_l = L.Linear(l_latent, 128 * 17 * l_seq // 32)
                self.deconv1 = L.Deconvolution2D(128, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
                self.deconv2 = L.Deconvolution2D(64, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
                self.deconv3 = L.Deconvolution2D(64, 64, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
                self.deconv4 = L.Deconvolution2D(64, 32, ksize=(4, 1), stride=(2, 1), pad=(1, 0))
                self.deconv5 = L.Deconvolution2D(32, 1, ksize=(4, 1), stride=(2, 1), pad=(1, 0))

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
            h = self.activate_func(h)
            if self.bn:
                h = self['dec_bn{}'.format(i)](h)
            h = self['deconv{}'.format(i)](h)
        return h
