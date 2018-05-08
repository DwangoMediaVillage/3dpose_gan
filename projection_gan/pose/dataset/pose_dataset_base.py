import chainer
import numpy as np


class Normalization(object):
    @staticmethod
    def normalize_3d(pose):
        # 3Dモデルの正規化
        # hip(0)と各関節点の距離の平均値が1になるようにスケール
        xs = pose.T[0::3] - pose.T[0]
        ys = pose.T[1::3] - pose.T[1]
        ls = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2)  # 原点からの距離 shape=(N-1,length)
        scale = ls.mean(axis=0)
        pose = pose.T / scale
        # hip(0)が原点になるようにシフト
        pose[0::3] -= pose[0].copy()
        pose[1::3] -= pose[1].copy()
        pose[2::3] -= pose[2].copy()
        return pose.T, scale

    @staticmethod
    def normalize_2d(pose):
        # 2DPoseの正規化
        # hip(0)と各関節点の距離の平均値が1になるようにスケール
        xs = pose.T[0::2] - pose.T[0]
        ys = pose.T[1::2] - pose.T[1]
        pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
        # hip(0)が原点になるようにシフト
        mu_x = pose[0].copy()
        mu_y = pose[1].copy()
        pose[0::2] -= mu_x
        pose[1::2] -= mu_y
        return pose.T


class PoseDatasetBase(chainer.dataset.DatasetMixin):
    def _normalize_3d(self, pose):
        return Normalization.normalize_3d(pose)

    def _normalize_2d(self, pose):
        return Normalization.normalize_2d(pose)
