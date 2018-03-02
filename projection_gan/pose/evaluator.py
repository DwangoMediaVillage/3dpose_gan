#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import copy

import chainer
from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer import reporter as reporter_module
from chainer.training import extensions


class Evaluator(extensions.Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        gen = self._targets['gen']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xy, xyz, scale = self.converter(batch, self.device)
                with function.no_backprop_mode(), \
                        chainer.using_config('train', False):
                    xy_real = xy
                    z_pred = gen(xy_real)
                    mse = F.mean_squared_error(z_pred, xyz[:, :, :, 2::3])
                    chainer.report({'mse': mse}, gen)

                    xx = gen.xp.power(xyz[:, :, :, 0::3] - xy[:, :, :, 0::2], 2)
                    yy = gen.xp.power(xyz[:, :, :, 1::3] - xy[:, :, :, 1::2], 2)
                    zz1 = gen.xp.power(xyz[:, :, :, 2::3] - z_pred.data, 2)
                    zz2 = gen.xp.power(xyz[:, :, :, 2::3] + z_pred.data, 2)

                    m1 = gen.xp.sqrt(xx + yy + zz1).mean(axis=3)[:, 0]
                    m2 = gen.xp.sqrt(xx + yy + zz2).mean(axis=3)[:, 0]
                    mae = gen.xp.where(m1 < m2, m1, m2)

                    m1 *= scale
                    mae *= scale
                    m1 = gen.xp.mean(m1)
                    mae = gen.xp.mean(mae)
                    chainer.report({'mae1': m1}, gen)
                    chainer.report({'mae2': mae}, gen)
            summary.add(observation)

        return summary.compute_mean()
