#!/bin/bash
save_dir="/mnt/work1/phat/TEST/DSMN/save_model/train_adap_car"
dataset="mskda_car"
net="vgg16"
pretrained_path="/mnt/work1/phat/TEST/DSMN/pre_trained_model/vgg16_caffe.pth"
max_epoch=20
burn_in=10

CUDA_VISIBLE_DEVICES=2 python train_msda.py --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --max_epoch ${max_epoch} --burn_in ${burn_in}\
 >train_msda_car.txt 2>&1

#13849
