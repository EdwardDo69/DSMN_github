from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pickle
import pprint
import sys
import time

import _init_paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.da_faster_rcnn.resnet import resnet
#from model.da_faster_rcnn.vgg16 import vgg16

#switch model
from msda.vgg16 import vgg16
#from last.vgg16_last_align import vgg16
# from model.faster_rcnn.vgg16 import vgg16
#from model.da_faster_rcnn.resnet import resnet

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import load_net, save_net, vis_detections
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch.autograd import Variable


print(sys.path)


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="cityscape",
        type=str,
    )
    parser.add_argument(
        "--num_epoch", dest="num_epoch", help="save res", default=-1, type=int,
    )
    parser.add_argument(
        "--output_dir", dest="output_dir", help="resoutput", default="./", type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default="cfgs/vgg16.yml",
        type=str,
    )
    parser.add_argument(
        "--net",
        dest="net",
        help="vgg16, res50, res101, res152",
        default="vgg16",
        type=str,
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # parser.add_argument('--load_dir', dest='load_dir',
    #                     help='directory to load models', default="models",
    #                     type=str)
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        help="directory to load models",
        default="models.pth",
        type=str,
    )
    parser.add_argument(
        "--part",
        dest="part",
        help="test_s or test_t or test_all",
        default="test_t",
        type=str,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true"
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )
    parser.add_argument(
        "--parallel_type",
        dest="parallel_type",
        help="which part of model to parallel, 0: all, 1: model before roi pooling",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--checksession",
        dest="checksession",
        help="checksession to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkepoch",
        dest="checkepoch",
        help="checkepoch to load network",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load network",
        default=10021,
        type=int,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="model file name",
        default="res101.bs1.pth",
        type=str,
    )
    parser.add_argument(
        "--vis", dest="vis", help="visualization mode", action="store_true"
    )

    parser.add_argument(
        "--USE_cls_cotrain",
        dest="USE_cls_cotrain",
        help="USE_cls_cotrain",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--USE_box_cotrain",
        dest="USE_box_cotrain",
        help="USE_box_cotrain",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--lc",
        dest="lc",
        help="whether use context vector for pixel level",
        action="store_true",
    )
    parser.add_argument(
        "--gc",
        dest="gc",
        help="whether use context vector for global level",
        action="store_true",
    )
    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--resume_name",
        dest="resume_name",
        help="resume checkpoint path",
        default="",
        type=str,
    )

    """
    args = parser.parse_args()
    """

    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[4,8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]

    elif args.dataset == "cityscape":
        print("loading our dataset...........")
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscape_2007_test_t"
        args.all_imdbtest_name = "cityscape_2007_test_all"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "mskda_bdd":
        print("loading our dataset...........")
        args.s1_imdb_name = "bdd100k_daytime_train"
        args.s2_imdb_name = "bdd100k_night_train"
        args.s1_imdbtest_name = "bdd100k_daytime_val"
        args.s2_imdbtest_name = "bdd100k_night_val"
        args.t_imdb_name = "bdd100k_dd_train"
        args.t_imdbtest_name = "bdd100k_dd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "water":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "water_train"
        args.t_imdbtest_name = "water_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "clipart":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_trainval"
        args.t_imdbtest_name = "clipart_trainval"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "comic":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "comic_train"
        args.t_imdbtest_name = "comic_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "bdd":
        print("loading our dataset...........")
        args.s_imdb_name = "citybdd7_train"
        args.t_imdb_name = "bdd_train"
        # args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "bdd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
        ]
    elif args.dataset == "sim10k":
        print("loading our dataset...........")
        args.s_imdb_name = "sim10k_2019_train"
        args.t_imdb_name = "cityscapes_car_2019_train"
        args.s_imdbtest_name = "sim10k_2019_val"
        args.t_imdbtest_name = "cityscapes_car_2019_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4,8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "mskda_car":
        print("loading our dataset...........")
        args.s1_imdb_name = "cityscapes_ms_car_train"
        args.s2_imdb_name = "KITTI_car_train"
        args.s1_imdbtest_name = "cityscapes_ms_car_test"
        args.s2_imdbtest_name = "KITTI_car_test"
        args.t_imdb_name = "bdd100k_daytime_car_train"
        args.t_imdbtest_name = "bdd100k_daytime_car_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    if args.part == "test_s":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(
            args.s_imdbtest_name, False
        )
    elif args.part == "test_t":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(
            args.t_imdbtest_name, False
        )
    elif args.part == "test_all":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(
            args.all_imdbtest_name, False
        )
    else:
        print("don't have the test part !")
        pdb.set_trace()

    imdb.competition_mode(on=True)

    print("{:d} roidb entries".format(len(roidb)))

    load_name = args.model_dir

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            imdb.classes,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=args.class_agnostic,

        )

    elif args.net == "res101":
        fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained_path=None,
            pretrained=False,
            class_agnostic=args.class_agnostic,
            lc=args.lc,
            gc=args.gc,
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        fasterRCNN = resnet(
            imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    # print(fasterRCNN.state_dict().keys())

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(
        {k: v for k, v in checkpoint["model_faster"].items() if k in fasterRCNN.state_dict()}
    )

    # fasterRCNN.load_state_dict(checkpoint['model'])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = args.part + args.model_dir.split("/")[-1]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(
        roidb,
        ratio_list,
        ratio_index,
        1,
        imdb.num_classes,
        training=False,
        normalize=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    data_iter = iter(dataloader)

    _t = {"im_detect": time.time(), "misc": time.time()}
    det_file = os.path.join(output_dir, "detections.pkl")

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()

        #arch1
        with torch.no_grad():
            (
                rois1,
                cls_prob1,
                bbox_pred1,
                rpn_loss_cls,
                rpn_loss_bbox,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                out_d_pixel_s1,
                out_d_s1,
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                subnet="subnet1"
            )

            (
                rois2,
                cls_prob2,
                bbox_pred2,
                rpn_loss_cls,
                rpn_loss_bbox,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                out_d_pixel_s2,
                out_d_s2,
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                subnet="subnet2"
            )

            (
                rois_ema,
                cls_prob_ema,
                bbox_pred_ema,
                rpn_loss_cls,
                rpn_loss_bbox,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                out_d_pixel_ema,
                out_d_ema,
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                subnet="ema"
            )

        scores1 = cls_prob1.data
        boxes1 = rois1.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas1 = bbox_pred1.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas1 = (
                        box_deltas1.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas1 = box_deltas1.view(1, -1, 4)
                else:
                    box_deltas1 = (
                        box_deltas1.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas1 = box_deltas1.view(1, -1, 4 * len(imdb.classes))

            pred_boxes1 = bbox_transform_inv(boxes1, box_deltas1, 1)
            pred_boxes1 = clip_boxes(pred_boxes1, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes1 = np.tile(boxes1, (1, scores1.shape[1]))

        pred_boxes1 /= data[1][0][2].item()

        scores1 = scores1.squeeze()
        pred_boxes1 = pred_boxes1.squeeze()

        scores2 = cls_prob2.data
        boxes2 = rois2.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas2 = bbox_pred2.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas2 = (
                            box_deltas2.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas2 = box_deltas2.view(1, -1, 4)
                else:
                    box_deltas2 = (
                            box_deltas2.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas2 = box_deltas2.view(1, -1, 4 * len(imdb.classes))

            pred_boxes2 = bbox_transform_inv(boxes2, box_deltas2, 1)
            pred_boxes2 = clip_boxes(pred_boxes2, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes2 = np.tile(boxes2, (1, scores2.shape[1]))

        pred_boxes2 /= data[1][0][2].item()

        scores2 = scores2.squeeze()
        pred_boxes2 = pred_boxes2.squeeze()

        scores_ema = cls_prob_ema.data
        boxes_ema = rois_ema.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas_ema = bbox_pred_ema.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas_ema = (
                            box_deltas_ema.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas_ema = box_deltas_ema.view(1, -1, 4)
                else:
                    box_deltas_ema = (
                            box_deltas_ema.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas_ema = box_deltas_ema.view(1, -1, 4 * len(imdb.classes))

            pred_boxes_ema = bbox_transform_inv(boxes_ema, box_deltas_ema, 1)
            pred_boxes_ema = clip_boxes(pred_boxes_ema, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes_ema = np.tile(boxes_ema, (1, scores_ema.shape[1]))

        pred_boxes_ema /= data[1][0][2].item()

        scores_ema = scores_ema.squeeze()
        pred_boxes_ema = pred_boxes_ema.squeeze()

        scores = torch.cat((scores1, scores2,scores_ema), dim=0)
        pred_boxes = torch.cat((pred_boxes1, pred_boxes2,pred_boxes_ema), dim=0)

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(
                        im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3
                    )
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack(
                [all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)]
            )
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write(
            "im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".format(
                i + 1, num_images, detect_time, nms_time
            )
        )
        sys.stdout.flush()

        if vis:
            cv2.imwrite("/mnt/work2/phat/TEST/conLoss_MSDAOD/result/result.png", im2show)
            cv2.waitKey(0)


    with open(det_file, "wb") as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print("Evaluating detections")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "eval_result.txt"), "a+") as ff:
        ff.write(str(args.num_epoch))
        ff.write("\n")
    imdb.evaluate_detections(all_boxes, args.output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
