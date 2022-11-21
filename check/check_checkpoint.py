import os
import torch

os.CUDA_VISIBLE_DEVICES = 3
start_epoch = 20
max_epoch = 20
output_dir= "/mnt/work2/phat/TEST/conLoss_MSDAOD/save_model/train_adap_car_2"

for epoch in range(start_epoch, max_epoch+1):
    resume_name = "mskda_car_{}.pth".format(epoch)
    load_name = os.path.join(output_dir, resume_name)
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    lmb1 = checkpoint["lmb1"]
    lmb2 = checkpoint["lmb2"]
    print( "Lmb1: {}, Lmb2: {}".format(len(lmb1), len(lmb2) ))
    if len(lmb1)>0 and len(lmb2)>0:
        print("Lmb1: {}, Lmb2: {}".format( sum(lmb1)/len(lmb1), sum(lmb2)/len(lmb2) ))
    print("loading finish %s" % (load_name))