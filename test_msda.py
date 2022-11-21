import os
net = "vgg16"
part = "test_t"
output_dir = "/mnt/work2/phat/TEST/conLoss_MSDAOD/output/combine_sw"
dataset = "mskda_bdd"
path = "/mnt/work2/phat/TEST/conLoss_MSDAOD/save_model/train_adap_sw_15con_5min"

start_epoch = 7
max_epochs = 16
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# all blue, sub1 green, sub2 red, ema purpule
for i in range(start_epoch, max_epochs + 1):
    model_dir = path + "/mskda_bdd_{}.pth".format(i)
    command = "python eval/test_msda.py --cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i)
    os.system(command)