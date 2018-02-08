#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import numpy as np
import chainer
import os
import glob
import h5py


class PoseDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root, action='all', length=32,
                 train=True, noise_scale=0):
        if train:
            data_set = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            data_set = ['S9', 'S11']

        with open('data/actions.txt') as f:
            actions_all = f.read().split('\n')[:-1]
        if action == 'all':
            actions = actions_all
        elif action in actions_all:
            actions = [action]
        else:
            raise Exception('Invalid action.')

        self.data_list = []
        self.npys = {}
        for action_name in actions:
            for dirname in data_set:
                npy_paths = glob.glob(os.path.join(
                    root, dirname, '{}_*.npy'.format(action_name)))
                npy_paths.sort()
                for npy_path in npy_paths:
                    basename = os.path.basename(npy_path)
                    action = os.path.splitext(basename)[0]
                    npy = np.load(npy_path)[::2]  # ダウンサンプリング
                    L = len(npy)

                    key = os.path.join(dirname, action)
                    self.npys[key] = npy

                    for start_pos in range(L - length + 1):
                        info = {'dirname': dirname, 'action': action,
                                'start_pos': start_pos, 'length': length}
                        self.data_list.append(info)
        self.train = train
        self.noise_scale = noise_scale

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
        scale = []
        # TODO(kogaki): forではなくarrayのまま処理できるように(CPU使いすぎ)
        for j in range(length):
            a = npy[start_pos + j].copy()

            # index=0の関節位置が(0,0,0)になるようにシフト
            a -= a[0]

            # index=0と8の関節位置の距離が1になるようにスケール変換
            body_length = np.sqrt(np.power(a[0] - a[8], 2).sum())
            a /= body_length
            scale.append(body_length)

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
                if self.train:
                    beta = np.random.randint(0, 4, 1) * np.pi / 2
                else:
                    beta = 0
            def rotY(arr):
                x, y, z = arr
                xx = x * np.cos(beta) + z * np.sin(beta)
                yy = y
                zz = -x * np.sin(beta) + z * np.cos(beta)
                return np.array([xx, yy, zz], dtype=arr.dtype)
            a4 = np.array(list(map(rotY, list(a3))))

            normalized_xyz.append(a4)
        normalized_xyz = np.array(normalized_xyz)
        xy = normalized_xyz[:, :, :2]
        xy = xy.reshape(length, -1)[None, :, :].astype(np.float32)

        z = normalized_xyz[:, :, 2]
        z = z.reshape(length, -1)[None, :, :].astype(np.float32)

        scale = np.array(scale, dtype=np.float32)

        if not self.train:
            np.random.seed(i)
        if self.noise_scale > 0:
            noise = np.random.normal(
                scale=self.noise_scale, size=xy.shape).astype(np.float32)
        else:
            noise = np.zeros(xy.shape, dtype=np.float32)
        noise = noise / scale[None, :, None]

        return xy, z, scale, noise


class SHDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root, action='all', length=32,
                 train=True, noise_scale=0):
        if train:
            data_set = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            data_set = ['S9', 'S11']

        with open('data/actions.txt') as f:
            actions_all = f.read().split('\n')[:-1]
        if action == 'all':
            actions = actions_all
        elif action in actions_all:
            actions = [action]
        else:
            raise Exception('Invalid action.')

        self.data_list = []
        self.npys = {}
        for action_name in actions:
            for dirname in data_set:
                h5_paths = glob.glob(os.path.join(
                    root, dirname, 'StackedHourglass', '{}*.h5'.format(
                        action_name[0].upper() + action_name[1:])))
                h5_paths.sort()
                for h5_path in h5_paths:
                    basename = os.path.basename(h5_path)
                    action = os.path.splitext(basename)[0]
                    npy = np.array(h5py.File(h5_path, 'r')['poses'])
                    npy = npy[np.power(npy[:, 6] - npy[:, 8], 2).sum(axis=1) != 0][::2] # ダウンサンプリング
                    L = len(npy)

                    key = os.path.join(dirname, action)
                    self.npys[key] = npy

                    for start_pos in range(L - length + 1):
                        info = {'dirname': dirname, 'action': action,
                                'start_pos': start_pos, 'length': length}
                        self.data_list.append(info)
        self.train = train
        self.noise_scale = noise_scale

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


        xy = []
        scale = []
        for j in range(length):
            pose = npy[start_pos + j].copy()

            x = pose[:, 0]
            y = pose[:, 1]
            body_length = np.sqrt(np.power(x[6] - x[8], 2) + np.power(y[6] - y[8], 2))
            if body_length == 0:
                raise Exception(x, y)
            scale.append(body_length)
            x /= body_length
            y /= body_length
            x -= x[6]
            y -= y[6]

            cos = -y[8] / (x[8]**2 + y[8]**2)
            sin = -x[8] / (x[8]**2 + y[8]**2)
            x = x * cos - y * sin
            y = x * sin + y * cos

            x = np.concatenate((np.zeros(1), x))
            y = np.concatenate((np.zeros(1), y)) * -1
            x[0] = x[9]
            y[0] = y[9]
            arr = []
            for i in [7, 3, 2, 1, 4, 5, 6, 8, 9, 0, 10, 14, 15, 16, 13, 12, 11]:
                arr.append(x[i])
                arr.append(y[i])

            xy.append(np.array(arr, np.float32))

        xy = np.array(xy)[None, :, :].astype(np.float32)
        z = np.zeros((1, length, xy.shape[2] // 2), np.float32)

        scale = np.array(scale, dtype=np.float32)

        if not self.train:
            np.random.seed(i)
        if self.noise_scale > 0:
            noise = np.random.normal(
                scale=self.noise_scale, size=xy.shape).astype(np.float32)
        else:
            noise = np.zeros(xy.shape, dtype=np.float32)
        noise = noise / scale[None, :, None]

        return xy, z, scale, noise
