# 2D Poseから3D PoseへProject

## 動作環境
  - Python 3.6.1
  - Chainer 2.1.0
  - Cupy 1.0.3
  - ProgressBar2 2.7.3 (動画作成時)
  - OpenCV 3.1.0 with ffmpeg (動画作成時)

## セットアップ
  - シンボリックリンクの作成
    ```
    git clone https://github.com/yasunorikudo/3d_pose.git
    cd 3d_pose
    git checkout dmv
    mkdir data
    ln -s /mnt/dataset/h36m/h3.6m/dataset data/h3.6m
    ```

## 学習
  - DCGAN
    ```
    python src/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt Adam \
      --bn f \
      --train_mode dcgan
    ```

  - DCGAN with Batch Normalization in Discriminator
    ```
    python src/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt Adam \
      --bn t \
      --train_mode dcgan
    ```

  - WGAN
    ```
    python src/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt RMSprop \
      --bn f \
      --train_mode wgan
    ```

  - Supervised Learning
    ```
    python src/train.py \
      --l_seq 64 \
      --gpu 0 \
      --epoch 500 \
      --opt Adam \
      --bn f \
      --train_mode supervised
    ```

## 学習結果の描画（動画の作成）
コマンド例
```
python src/create_video.py results/2017-12-08_06-33-26/gen_epoch_1.npz
```
