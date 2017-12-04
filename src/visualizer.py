#!/usr/bin/env python

import os

import copy
import cv2
import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import cuda
from chainer import Variable

from dataset_foreground import LabeledImageDataset

def out_image(args, updater, enc, dec, mean, rows, cols):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        h, w = 224, 224

        np.random.seed(args.seed)
        test_d = LabeledImageDataset(args.test, args.img_root, args.seg_root,
                                     224, args.n_class, mean, random=False)
        iterator = chainer.iterators.SerialIterator(test_d, cols,
                                                    repeat=False, shuffle=True)

        img_in = np.empty((rows, cols, h, w, 3), dtype=np.uint8)
        img_out = np.empty((rows, cols, h, w, 3), dtype=np.uint8)
        saliency = np.empty((rows, cols, h, w), dtype=np.uint8)

        for i in range(rows):
            batch = iterator.next()
            device = updater.device
            x_in, t, seg = chainer.dataset.concat_examples(
                batch, device=device)

            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                x_in = Variable(x_in)
                x_noise = dec(enc(x_in))
                x_out = x_in + x_noise

            orig_in = cuda.to_cpu(x_in.data) + mean
            img_in[i] = orig_in.transpose(0, 2, 3, 1).astype(np.uint8)

            def clip(x):
                if x > 255:
                    return 255
                elif x < 0:
                    return 0
                else:
                    return x
            vf = np.vectorize(clip)
            orig_out = cuda.to_cpu(x_out.data) + mean
            orig_out = vf(orig_out)
            img_out[i] = orig_out.transpose(0, 2, 3, 1).astype(np.uint8)

            noise = np.abs(cuda.to_cpu(x_noise.data)).mean(axis=1).transpose(1, 2, 0)
            noise -= noise.min(axis=(0, 1))
            noise /= noise.max(axis=(0, 1))
            noise *= 255.
            noise = noise.astype(np.uint8).transpose(2, 0, 1)
            saliency[i] = noise

        img_in = img_in.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, 3)
        img_out = img_out.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, 3)
        saliency = saliency.transpose(0, 2, 1, 3).reshape(rows * h, cols * w)
        saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

        img_dir = os.path.join(args.dir, 'images')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        cv2.imwrite(os.path.join(img_dir, 'in_{}_epoch.png'.format(trainer.updater.epoch)), img_in)
        cv2.imwrite(os.path.join(img_dir, 'out_{}_epoch.png'.format(trainer.updater.epoch)), img_out)
        cv2.imwrite(os.path.join(img_dir, 'saliency_{}_epoch.png'.format(trainer.updater.epoch)), saliency)
        iterator.reset()

    return make_image
