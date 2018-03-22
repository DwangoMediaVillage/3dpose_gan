# 2D Poseから3D PoseへProject


## Run Inference (with openpose)

1. Download openpose pretrained model
    * openpose_pose_coco.prototxt
        * https://github.com/opencv/opencv_extra/blob/3.4.1/testdata/dnn/openpose_pose_coco.prototxt
    * pose_iter_440000.caffemodel
        * http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
2. Download 3dpose_gan pretrained model
    * gen_epoch_500.npz
        * TBA
3. Run Inference
    * ` python bin/demo.py your/image.png --lift_model gen_epoch_500.npz --model pose_iter_440000.caffemodel --proto2d pose_deploy_linevec.prototxt`
    * **Need OpenCV >= 3.4**
        * < 3.3 results extreamly wrong estimation

## 動作環境
  - Python 3.6.1
  - Cupy 2.2.0
  - Chainer 3.2.0
  - ChainerUI 0.1.0
  - tqdm 4.19.5 (動画作成時)
  - OpenCV 3.4 with ffmpeg (動画作成時)

## セットアップ
  - データセットのダウンロード
    ```
    git clone https://github.com/DwangoMediaVillage/3dpose_gan.git
    cd 3dpose_gan
    cd data
    wget https://www.dropbox.com/s/dgtpcudkm1jndh3/h3.6m.tar.gz
    tar zxvf h3.6m.tar.gz
    rm h3.6m.tar.gz
    wget https://www.dropbox.com/s/f8fn0midnswvo03/points.tar.gz
    tar zxvf points.tar.gz
    rm points.tar.gz
    wget https://www.dropbox.com/s/n8344hwlc1lwxiw/sh_detect.pickle
    cd ..
    ```

## 学習
  - DCGAN
    ```
    python bin/train.py \
      --l_seq 1 \
      --gpu 0 \
      --epoch 500 \
      --opt Adam \
      --bn f \
      --train_mode dcgan \
      --dcgan_accuracy_cap 0.9 \
      --action all \
      --nn linear \
      --noise_scale 0
    ```

  - Supervised Learning
    ```
    python bin/train.py \             
      --l_seq 1 \
      --gpu 2 \
      --epoch 500 \
      --opt Adam \
      --bn t \
      --act_func relu \
      --train_mode supervised \
      --nn linear \
      --noise_scale 0
    ```

#### 以下検証中...
  - DCGAN with Batch Normalization in Discriminator
    ```
    python bin/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt Adam \
      --bn t \
      --train_mode dcgan
    ```

  - WGAN
    ```
    python bin/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt RMSprop \
      --bn f \
      --train_mode wgan
    ```

## 学習結果の描画（動画の作成）
コマンド例
```
python bin/create_video.py results/2017-12-08_06-33-26/gen_epoch_1.npz
```
最初の1フレームを静止画で保存
```
python bin/create_video.py results/2017-12-08_06-33-26/gen_epoch_1.npz --image
```

## 詳細評価
コマンド例
```
python bin/eval.py results/2017-12-08_06-33-26/gen_epoch_1.npz
```
