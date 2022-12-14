# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import, division, print_function

# from scipy.misc import imread
import cv2
import numpy as np
import numpy.random as npr
from model.utils.blob import im_list_to_blob, prep_im_for_blob
from model.utils.config import cfg

from torchvision import transforms
from Augment.augmentation_impl import GaussianBlur


def get_minibatch_strongaug(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert (
        cfg.TRAIN.BATCH_SIZE % num_images == 0
    ), "num_images ({}) must divide BATCH_SIZE ({})".format(
        num_images, cfg.TRAIN.BATCH_SIZE
    )

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {"data": im_blob}

    im_name = roidb[0]["image"]
    if im_name.find("source_") == -1:  # target domain
        blobs["need_backprop"] = np.zeros((1,), dtype=np.float32)
    else:
        blobs["need_backprop"] = np.ones((1,), dtype=np.float32)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]["gt_classes"] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(
            (roidb[0]["gt_classes"] != 0)
            & np.all(roidb[0]["gt_overlaps"].toarray() > -1.0, axis=1)
        )[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]["boxes"][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]["gt_classes"][gt_inds]
    blobs["gt_boxes"] = gt_boxes
    blobs["im_info"] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
    )

    blobs["img_id"] = roidb[0]["img_id"]

    # change gt_classes to one hot
    def gt_classes2cls_lb_onehot(array):
        cls_lb = np.zeros((num_classes - 1,), np.float32)
        for i in array:
            cls_lb[i - 1] = 1
        return cls_lb

    blobs["cls_lb"] = gt_classes2cls_lb_onehot(roidb[0]["gt_classes"])

    return blobs




def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []

    # _________________________________________________________________________________
    # StrongAgmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]

        ),
    ])
    # _________________________________________________________________________________

    for i in range(num_images):
        im = cv2.imread(roidb[i]["image"])
        #_________________________________________________________________________________
        im = transform(im)
        im = np.asarray(im)

        # _________________________________________________________________________________
        #cv2.imwrite('/mnt/work2/phat/CODING/crosscam_conLoss/test/test{}.jpg'.format(i), im)
        # im = imread(roidb[i]["image"])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]

        if roidb[i]["flipped"]:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
