from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import csv
import os
import sys

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

def parse(value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)), None)

def open_for_csv(path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

def read_annotations(csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise(ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                           None)

            x1 = parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                # raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                continue
            if y2 <= y1:
                continue
                # raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
            # check if the current class name is correctly present
            if class_name not in classes:
                continue
                # raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result
def name_to_label(name, classes):
        return classes[name]
def load_annotations(image_index, classes):
    # get ground truth annotations
    annotation_list = image_data[image_names[image_index]]
    annotations = np.zeros((0, 5))

    # some images appear to miss annotations (like image with id 257034)
    if len(annotation_list) == 0:
        return None

    # parse annotations
    for idx, a in enumerate(annotation_list):
        # some annotations have basically no width / height, skip them
        x1 = a['x1']
        x2 = a['x2']
        y1 = a['y1']
        y2 = a['y2']

        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue

        annotation = np.zeros((1, 5))

        annotation[0, 0] = x1
        annotation[0, 1] = y1
        annotation[0, 2] = x2
        annotation[0, 3] = y2

        annotation[0, 4] = name_to_label(a['class'], classes)
        annotations = np.append(annotations, annotation, axis=0)

    return annotations

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Modify the path to your own
data_dir = "/root/data1/dataset/DSEC"


datasets = ["all", "sub"]
for dataset in datasets:
    annotation_dir = os.path.join(data_dir, "annotations",dataset)
    os.makedirs(annotation_dir, exist_ok=True)
    label_root = os.path.join('datasets', dataset)
    label_files = os.listdir(label_root)
    for label_file in label_files:
        label_path = os.path.join(label_root, label_file)
        print(label_path)

        if "2classes" in label_file:
            cats = ['car', 'pedestrian']
            class_list = './datasets/labels_2classes.csv'
        else:
            cats = ['car', 'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'truck', 'bus', 'train']
            class_list = './datasets/labels.csv'
        cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
        cat_info = []
        for i, cat in enumerate(cats):
            cat_info.append({'name': cat, 'id': i + 1})

        with open_for_csv(class_list) as file:
            classes = load_classes(csv.reader(file, delimiter=','))

        with open_for_csv(label_path) as file:
            image_data = read_annotations(csv.reader(file, delimiter=','), classes)

        image_names = list(image_data.keys())
        ret = {'images': [], 'annotations': [], "categories": cat_info}

        count1 = 1
        count2 = 1
        for i in range(len(image_names)):
            file = image_names[i]
            annot = load_annotations(i, classes)
            if annot is None:
                continue
            image_info = {'file_name': image_names[i].replace('npz', 'png').replace('left', 'images/left/transformed'),
                        'id': count1,
                        'height': 480,
                        'width': 640}
            ret['images'].append(image_info)

            for ann in annot:
                cat_id = ann[4]
                bbox = [float(ann[0]), float(ann[1]), float(ann[2]), float(ann[3])]

                ann = {'image_id': count1, 
                        'id': count2,
                        'category_id': int(cat_id)+1,
                        'bbox': _bbox_to_coco_bbox(bbox),
                        'iscrowd': 0,
                        'area': ((float(ann[2])-float(ann[0]))*(float(ann[3])-float(ann[1])))
                        }
                
                count2 = count2 + 1
                ret['annotations'].append(ann)
            count1 = count1 + 1

        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        out_path = os.path.join(annotation_dir, label_file.replace('csv', 'json'))
        json.dump(ret, open(out_path, 'w'))
