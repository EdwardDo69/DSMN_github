#!/bin/bash
save_dir="/mnt/work1/phat/DSMN_github/save_model/bdd"
dataset="mskda_bdd"
net="vgg16"
pretrained_path="/mnt/work1/phat/DSMN_github/pre_trained_model/vgg16_caffe.pth"
max_epoch=25
burn_in=10

CUDA_VISIBLE_DEVICES=3 python train_msda.py --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --max_epoch ${max_epoch} --burn_in ${burn_in}\
 >train_msda_bdd.txt 2>&1

#13849
