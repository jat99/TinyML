"""
Author: Jose Torres
Date: August 19, 2024
Description: 
Filename:
"""

import os
import xml.etree.ElementTree as ET
import pickle
import cv2

class DataGenerator():

    def __init__(self):
        self.voc_dataset_path = ''
        self.annotations_dir = 'Annotations/'
        self.images_dir = 'JPEGImages/'
        self.train_dir = 'ImageSets/Main/train.txt'
        self.val_dir = 'ImageSets/Main/val.txt'
        self.voc_classes = [
            "background","aeroplane","bicycle","bird","boat","bottle",
            "bus","car","cat","chair","cow","diningtable","dog","horse",
            "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
        ]

    def get_image_path(self, image_id):
        return os.path.join(self.voc_dataset_path, self.images_dir,image_id + '.jpg')

    def get_train_image_ids(self):
        with open(self.voc_dataset_path + self.train_dir, 'r') as file:  
            train_image_ids = [line.strip() for line in file]

        return train_image_ids
    
    def get_val_image_ids(self):
        with open(self.voc_dataset_path + self.val_dir, 'r') as file:
            val_image_ids = [line.strip() for line in file]

        return val_image_ids
    
    def parse_xml(self, image_ids=[]):
        voc_data_set = []
        for id in image_ids:
            tree = ET.parse(self.voc_dataset_path + self.annotations_dir + id + '.xml')
            root = tree.getroot()
            try:
                image_dimensions = (root[3][0].text, root[3][1].text, root[3][2].text)
                image_data = VOC_DataSet_Image(img_id=id,img_size=image_dimensions)
            except (IndexError, TypeError, ValueError) as e:
                continue

            classes = []
            
            for object in root.iter('object'):
                try: 
                    class_name = object[0].text
                    xmin = object[4][0].text
                    ymin = object[4][1].text
                    xmax = object[4][2].text
                    ymax = object[4][3].text
                    coordinates = (xmin, ymin, xmax, ymax)
                    image_data.add_object(class_name=class_name, coordinates=coordinates)
                except (IndexError, TypeError, ValueError) as e:
                    continue
            voc_data_set.append(image_data)
        return voc_data_set
    
    def get_jpeg_images():
        pass
    
    def get_train_dataset(self):
        with open('train_dataset.pickle', 'rb') as file:
            train_dataset: VOC_DataSet = pickle.load(file)

        images = []
        total_detections = []
        for set in train_dataset.dataset:
            img = cv2.imread(self.get_image_path(set.img_id))
            img = cv2.resize(img, (300,300))
            images.append(img)
            detections = []
            for key in set.objects.keys():
                for coordinates in set.objects[key]:
                    label = int(self.voc_classes.index(key))
                    xmin = int(coordinates[0])
                    ymin = int(coordinates[1])
                    xmax = int(coordinates[2])
                    ymax = int(coordinates[3])
                    detections.append([label, xmin,ymin, xmax, ymax])
            total_detections.append(detections)
        return (images, total_detections)
    
class VOC_DataSet():
    def __init__(self, parsed_data):
        self.dataset = parsed_data

    def get_img_id(self):
        return self.dataset[0].img_id
    
    def get_img_size(self):
        return self.dataset[0].img_size
    
    def get_objects(self):
        return self.dataset[0].objects
    
class VOC_DataSet_Image():
    def __init__(self, img_id, img_size):
        self.img_id = img_id
        self.img_size = img_size
        self.objects = {}

    def add_object(self, class_name, coordinates):
        if class_name in self.objects:
            self.objects[class_name].append(coordinates)
        else:
            self.objects[class_name] = [coordinates]
