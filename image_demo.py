#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_96_coco.pb"
image_path      = "img/test2.jpg"
min_score       = 0.7
num_classes     = 1
input_size      = 544
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

#bboxes = utils.nms(pred_bbox, 0.45, method='soft-nms')
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.6)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes, min_score)
#image = Image.fromarray(image).resize((2666,2000),Image.ANTIALIAS)
#image = cv2.resize(image,(2666,2000))

totalNum = 0
sureNum = 0
unsureNum = 0
for i, bbox in enumerate(bboxes):
    score = bbox[4]
    totalNum+=1
    if score>min_score:
        sureNum+=1
    else:
        unsureNum+=1

print('钢筋数量：合计%d \t高准确度%d \t低准确度%d' % (totalNum, sureNum, unsureNum))

image = Image.fromarray(image)
image.show()
image.save('test_mark.jpg')




