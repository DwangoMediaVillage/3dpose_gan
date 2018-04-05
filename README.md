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
    * ` python bin/demo.py your/image.png --lift_model sample/gen_epoch_500.npz --model pose_iter_440000.caffemodel --proto2d pose_deploy_linevec.prototxt`
    * **Need OpenCV >= 3.4**
        * < 3.3 results extreamly wrong estimation

## Dependencies(Recommended versions)
  - Python 3.6.1
  - Cupy 2.2.0
  - Chainer 3.2.0
  - ChainerUI 0.1.0
  - tqdm 4.19.5
  - OpenCV 3.4 with ffmpeg
  - git-lfs
    - to download pre-trained model
    - or you can download pre-trained model directory from [https://github.com/DwangoMediaVillage/3dpose_gan/blob/master/sample/gen_epoch_500.npz?raw=true](https://github.com/DwangoMediaVillage/3dpose_gan/blob/master/sample/gen_epoch_500.npz?raw=true)

## Training

TBA

## Evaluation

TBA
