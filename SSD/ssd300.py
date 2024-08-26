"""
Author: Jose Torres
Date: 
Description:
Filename: ssd300.py
"""
from ssd import ssd

class SSD300:

    def __init__(self):
        self.feature_maps = [(19, 19), (10, 10),(5, 5),(3, 3),(2, 2),(1, 1)]
        self.scales = [0.2, 0.35, 0.5, 0.75, 0.95]
        self.aspect_ratios = [0.5, 1, 2]
        self.input_shape = (300,300,3)
        self.voc_classes = [
            "background","aeroplane","bicycle","bird","boat","bottle",
            "bus","car","cat","chair","cow","diningtable","dog","horse",
            "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
        ]
        self.ssd = ssd(
                    input_shape=self.input_shape,
                    classes=self.voc_classes,
                    feature_maps=self.feature_maps, 
                    scales=self.scales,
                    aspect_ratios=self.aspect_ratios
                    )

    def add_base_network():
        pass

ssd_300 = SSD300()

anchor_boxes = [
    [0.2, 0.3, 0.1, 0.1],
    [0.15, 0.15, 0.4, 0.5],
    [0.7, 0.2, 0.15, 0.3],
    [0.4, 0.7, 0.25, 0.2],
    [0.6, 0.4, 0.1, 0.15],
    [0.3, 0.6, 0.2, 0.1],
    [0.8, 0.5, 0.3, 0.3],
    [0.1, 0.1, 0.05, 0.05],
    [0.7, 0.7, 0.1, 0.2],
    [0.5, 0.3, 0.15, 0.15],
    ]

gt_boxes = [
    [0.15, 0.25, 0.25, 0.35],
    [0.4, 0.45, 0.6, 0.65],
    [0.6, 0.1, 0.75, 0.4],
    [0.3, 0.55, 0.55, 0.75],
    [0.5, 0.35, 0.65, 0.5],
    [0.25, 0.5, 0.45, 0.6],
    [0.65, 0.4, 0.95, 0.7],
    [0.05, 0.05, 0.1, 0.1],
    [0.6, 0.55, 0.75, 0.75],
    [0.4, 0.25, 0.55, 0.4],
]

# Base Network

# Compile

# Fit

# Evaluate

# Predict 




