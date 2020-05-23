## Unofficial pytorch implementation: Attentive Generative Adversarial Network for Raindrop Removal from A Single Image (CVPR'2018)

Reference: https://github.com/rui1996/DeRaindrop

## Description:
Fork from https://github.com/shleecs/DeRaindrop_unofficial .
Try to change the GPUs' problems. 

## Prerequisites:
1. Linux
2. Python 3.7
3. NVIDIA GPU + CUDA CuDNN (CUDA 10.0)
4. Pytorch 1.4.1
5. TensorboardX

## Train

```
python train.py --gpu 0,1 --train_dataset ../dataset/Derain/train/train/ --eval_dataset ../dataset/Derain/test_a/test_a/ ../dataset/Derain/test_b/test_b/ --batch_size 4
```

## Demo 

The demo pictures are put under the directory `./demo/input/` and `./demo/output/` is a sample of the ouput of the model. If you want to generate your own, use the following code:

```
CUDA_VISIBLE_DEVICES=gpu_id python predict.py --mode demo --input_dir ./demo/input/ --output_dir ./demo/output/
```
*Please change the line 60 of predict.py to your own G.pth dir*

## Additional implemented parts:

Training source / loss implementation / dataloader ... 

## Dataset
The whole dataset can be find in author pages(https://github.com/rui1996/DeRaindrop)

## Results

