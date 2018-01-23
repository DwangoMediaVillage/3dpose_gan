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
                xy, z, scale = self.converter(batch, self.device)
                with function.no_backprop_mode(), \
                        chainer.using_config('train', False):
                    z_pred = gen(xy)
                    mse = F.mean_squared_error(z_pred, z)
                    chainer.report({'mse': mse}, gen)
                    m1 = gen.xp.abs(z - z_pred.data).mean(axis=3)[:, 0]
                    m2 = gen.xp.abs(z + z_pred.data).mean(axis=3)[:, 0]
                    mae = gen.xp.where(m1 < m2, m1, m2)
                    m1 *= scale
                    mae *= scale
                    m1 = gen.xp.mean(m1)
                    mae = gen.xp.mean(mae)
                    chainer.report({'mae1': m1}, gen)
                    chainer.report({'mae2': mae}, gen)
            summary.add(observation)

        return summary.compute_mean()
