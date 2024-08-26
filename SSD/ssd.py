"""
Author: Jose Torres
Description: Single Shot Detector Custom Functions 
Date: 8/17/2024
Filename: ssd.py
"""
from tensorflow.python.keras.layers import Conv2D, Reshape, Concatenate, Activation
import numpy as np

class ssd:

    def __init__(
            self,
            input_shape, # height & width
            classes = [], # include background ['background']
            feature_maps = [], 
            scales = [], 
            aspect_ratios = []
            ):
        self.input_shape = input_shape
        self.num_classes = len(classes)
        self.classes = classes
        self.feature_maps = feature_maps
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_boxes = self.create_anchors()
        self.num_anchor_boxes = len(self.anchor_boxes)

    def confidence_predictor(self, x):
        filter_amount = self.num_anchor_boxes * (self.num_classes + 1)
        output = Conv2D(filters=filter_amount, kernel_size=3, padding=1)(x)
        return x,output

    def bbox_predictor(self, x):
        filter_amount = self.num_anchor_boxes * 4
        output = Conv2D(filters=filter_amount, kernel_size=3, padding=1)(x)
        return x, output
    
    # generate anchor boxes for all feature maps 
    def create_anchors(self):
        anchors = []
        """
        On each feature map, a certain of amount of anchors are 
        generated per pixel, @ the center.
        """
        for feature_map in self.feature_maps: # Each feature map generates 
           for i in range(feature_map[0]):
               for j in range(feature_map[1]):
                    # Determine the center points 
                    x = (i + 0.5) / feature_map[0]
                    y = (j + 0.5) / feature_map[1]
                    
                    for scale in self.scales:
                        for ratio in self.aspect_ratios:
                            # Maintain constant area across differnt aspect ratios!
                            h = scale / np.sqrt(ratio)
                            w = scale * np.sqrt(ratio)
                            anchors.append([x, y, w, h])

        return np.array(anchors)
    # calculate confidence and localization loss
    def ssd_loss(self, y_true, y_pred):
        """ 
        Localization: Positive matches only

        """
        """
        Confidence: Positive & Negative
        Positive: 
        Negative: 
        """

        masked_boxes = self.match_boxes(y_true, y_pred)
        conf_loss = 0
        loc_loss = 0

        return conf_loss + loc_loss

    # returns value 0 - 1, intersection over union
    def iou(self, box_one, box_two):
        """
        parameters: box_one = (x1, y1, x2, y2)
        """

        xmin1, ymin1 = box_one[0], box_one[1]
        xmax1, ymax1 = box_one[2], box_one[3]
        xmin2, ymin2 = box_two[0], box_two[1]
        xmax2, ymax2 = box_two[2], box_two[3]

        top_left = (max(xmin1, xmin2), max(ymin1, ymin2))
        bottom_right = (min(xmax1, xmax2), min(ymax1, ymax2))

        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        intersection_area = max(0,width) * max(0, height)

        box_one_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        box_two_area = (xmax2 - xmin2) * (ymax2 - ymin2)

        union = box_one_area + box_two_area - intersection_area

        try:
            return intersection_area / union
        except ZeroDivisionError:
            return 1

    # used to match all anchors with ground truth boxes
    # when anchor boxes are accurately predicted
    def match_boxes(self, y_true, y_pred, iou_threshold = 0.5):
        """
        1. Adjust appropriate anchor box (Add offsets)
        2. Match @ least one anchor to gt box w/ highest iou
        3. Match all that meet certain threshold
        """
        anchor_mask = []

        for j in range(len(y_pred)): # Ground Truth
            sub_array = []
            for i in range(len(y_true)): # Default (x, y, w, h)
                xmin = y_pred[j][0] - y_pred[j][2] / 2
                ymin = y_pred[j][1] - y_pred[j][3] / 2
                xmax = y_pred[j][0] + y_pred[j][2] / 2
                ymax = y_pred[j][1] + y_pred[j][3] / 2
                box = (xmin, ymin, xmax, ymax)
                iou = self.iou(box, y_true[i])
                mask = -1 
                if iou > iou_threshold:
                    mask = i
                sub_array.append(mask)
            anchor_mask.append(sub_array)

        return np.array(anchor_mask)
    
    def non_maximum_supression():
        """
        From predictions, select only anchor with highest
        """
        pass