# Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations

This is the authors' implementation of [Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations
](https://arxiv.org/abs/1803.08244)

![](https://nico-opendata.jp/assets/img/casestudy/3dpose_gan/system.png)

![](https://nico-opendata.jp/assets/img/casestudy/3dpose_gan/mpii.jpg)

## Run Inference for demo (with openpose)

1. Download openpose pretrained model
    * openpose_pose_coco.prototxt
        * https://github.com/opencv/opencv_extra/blob/3.4.1/testdata/dnn/openpose_pose_coco.prototxt
    * pose_iter_440000.caffemodel
        * http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
2. Run Inference
    * ` python bin/demo.py sample/image.png --lift_model sample/gen_epoch_500.npz --model2d pose_iter_440000.caffemodel --proto2d openpose_pose_coco.prototxt`
    * **Need OpenCV >= 3.4**
        * < 3.3 results extreamly wrong estimation

## Dependencies(Recommended versions)
  - Python 3.6.5
  - Cupy 4.0.0
  - Chainer 4.0.0
  - OpenCV 3.4 (when showing results)
  - git-lfs
    - to download pre-trained model
    - or you can download pre-trained model directory from [https://github.com/DwangoMediaVillage/3dpose_gan/blob/master/sample/gen_epoch_500.npz?raw=true](https://github.com/DwangoMediaVillage/3dpose_gan/blob/master/sample/gen_epoch_500.npz?raw=true)

## Training
#### Human3.6M dataset
  - [x] Unsupervised learning of 3D points from ground truth 2D points

    ```
    python bin/train.py --gpu 0 --mode unsupervised --dataset h36m --use_heuristic_loss
    ```
  - [ ] Unsupervised learning of 3D points from detected 2D points by Stacked Hourglass

    TBA

  - [x] Supervised learning of 3D points from ground truth 2D points

    ```
    python bin/train.py --gpu 0 --mode supervised --activate_func relu --use_bn
    ```

#### MPII dataset
TBA

#### MPI-INF-3DHP dataset
TBA

## Evaluation
```
python bin/eval.py results/hoge/gen_epoch_*.npz
```
