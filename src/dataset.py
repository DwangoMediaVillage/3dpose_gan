#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import numpy as np
import chainer
import os
import glob


class PoseDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root, length=32, train=True):
        if train:
            data_set = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            data_set = ['S5']

        self.data_list = []
        self.npys = {}
        for dirname in data_set:
            npy_paths = glob.glob(os.path.join(root, dirname, 'walking_*.npy'))
            npy_paths.sort()
            for npy_path in npy_paths:
                basename = os.path.basename(npy_path)
                action = os.path.splitext(basename)[0]
                npy = np.load(npy_path)[::2]  # ここ修正する？
                L = len(npy)

                key = os.path.join(dirname, action)
                self.npys[key] = npy

                for start_pos in range(L - length + 1):
                    info = {'dirname': dirname, 'action': action,
                            'start_pos': start_pos, 'length': length}
                    self.data_list.append(info)
        self.train = train

    def __len__(self):
        return len(self.data_list)

    def get_example(self, i):
        info = self.data_list[i]
        dirname = info['dirname']
        action = info['action']
        start_pos = info['start_pos']
        length = info['length']

        key = os.path.join(dirname, action)
        npy = self.npys[key]

        normalized_xyz = []
        for j in range(length):
            a = npy[start_pos + j].copy()

            # index=0の関節位置が(0,0,0)になるようにシフト
            a -= a[0]

            # index=0と8の関節位置の距離が1になるようにスケール変換
            body_length = np.sqrt(np.power(a[0] - a[8], 2).sum())
            a /= body_length

            # z軸回りの回転（index=8の関節位置がx=0となるように）
            x, y = a[8, :2]
            cos_gamma = y / np.sqrt(x**2 + y**2)
            sin_gamma = x / np.sqrt(x**2 + y**2)
            def rotZ(arr):
                x, y, z = arr
                xx = x * cos_gamma - y * sin_gamma
                yy = x * sin_gamma + y * cos_gamma
                zz = z
                return np.array([xx, yy, zz], dtype=arr.dtype)
            a2 = np.array(list(map(rotZ, list(a))))

            # x軸回りの回転（index=8の関節位置がy軸上に来るように）
            y2, z2 = a2[8, 1:]
            cos_alpha = y2 / np.sqrt(y2**2 + z2**2)
            sin_alpha = -z2 / np.sqrt(y2**2 + z2**2)
            def rotX(arr):
                x, y, z = arr
                xx = x
                yy = y * cos_alpha - z * sin_alpha
                zz = y * sin_alpha + z * cos_alpha
                return np.array([xx, yy, zz], dtype=arr.dtype)
            a3 = np.array(list(map(rotX, list(a2))))

            # y軸回りの回転（回転角はrandomに決定）
            if j == 0:
                if not self.train:
                    np.random.seed(i)
                beta = np.random.uniform(0, 2 * np.pi)
            def rotY(arr):
                x, y, z = arr
                xx = x * np.cos(beta) + z * np.sin(beta)
                yy = y
                zz = -x * np.sin(beta) + z * np.cos(beta)
                return np.array([xx, yy, zz], dtype=arr.dtype)
            a4 = np.array(list(map(rotY, list(a3))))

            normalized_xyz.append(a4)
        normalized_xyz = np.array(normalized_xyz)
        xy = normalized_xyz[:, :, :2].transpose(0, 2, 1)
        xy = xy.reshape(length, -1)[None, :, :].astype(np.float32)
        z = normalized_xyz[:, :, 2]
        z = z.reshape(length, -1)[None, :, :].astype(np.float32)
        return xy, z
