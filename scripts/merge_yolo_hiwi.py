#!/usr/bin/env python3
"""A script to merge hiwi labels and all yolo rounds labels
the idea is that each round will have different class numbers
should be ran after inference of round x, in preparation for the labelling QC on CVAT

eg:
hiwi labelled honeybee: 0
hiwi labelled bumblebee: 1
hiwi labelled unknown_bee: 2

round1-yolo-pred(+linn corrected) labelled honeybee: 3
round1-yolo-pred(+linn corrected) labelled bumblebee: 4
round1-yolo-pred(+linn corrected) labelled unknown_bee: 5
round1-yolo-pred(high iou) labelled honeybee: 6
round1-yolo-pred(high iou) labelled bumblebee: 7
round1-yolo-pred(high iou) labelled unknown_bee: 8

round2-yolo-pred(+linn corrected) labelled honeybee: 9
round2-yolo-pred(+linn corrected) labelled bumblebee: 10
round2-yolo-pred(+linn corrected) labelled unknown_bee: 11
round2-yolo-pred(high iou) labelled honeybee: 12
round2-yolo-pred(high iou) labelled bumblebee: 13
round2-yolo-pred(high iou) labelled unknown_bee: 14
"""

import click
from tqdm import tqdm 
import os
import pdb
import shutil
import glob
import torch
import torchvision
from PIL import Image
from pathlib import Path  # python 3.4+
from scipy.optimize import linear_sum_assignment
import numpy as np


def labels2list(file_f):
    annos_og = file_f.readlines()

    # remove whitespaces
    annos = [anno.translate(str.maketrans('', '', '\n\t\r')) for anno in annos_og]
    annos = [anno.split(' ') for anno in annos]
    return annos 


def get_cost(obj1, obj2):
    iou = obj1.get_iou(obj2)
    cost = 1.0 - iou
    return cost 

def get_cost_row(obj1, obj_list):
    cost_row = []
    for obj2 in obj_list:
        cost_row.append(get_cost(obj1, obj2))
    return cost_row # this could be done with list comprehension but eh.

def find_matches(labels_list, pred_list, min_iou):
    """
    also updates the list of pred objs that the match is found
    """
    num_preds = len(pred_list)
    num_labels = len(labels_list)
    cost_mat = np.zeros((num_preds, num_labels))
    for pred_i, pred in enumerate(pred_list):
        cost_mat[pred_i] = get_cost_row(pred, labels_list) 

    rows, cols = linear_sum_assignment(cost_mat)
    matches = []

    for r, c in zip(rows, cols):
        iou = cost_mat[r,c]
        if iou <= min_iou:
            matches.append((r,c))

    for match in matches:
        pred_i = match[0]
        pred_list[pred_i].is_pred_and_correct = True

    return matches


class id_mapper:
    """Maps class ids from the usual to that of the round
    ids are always strings 
    """
    def __init__(self, round_i, num_classes=3):
        self.round_i = round_i
        self.num_classes = num_classes
        
    def get_new_id(self, old_id, is_corr_pred=False):
        if is_corr_pred:  # 6, 7, 8
            new_id = (2 * self.round_i) * (self.num_classes) + int(old_id)
        else:  # 3, 4, 5
            new_id = (2 * self.round_i - 1) * (self.num_classes) + int(old_id)
        return new_id

    def convert_obj_line(self, obj_line_list):
        new_line_list = list(obj_line_list)
        new_line_list[0] = self.get_new_id(obj_line_list[0])
        
        # remove confidence thing 
        new_line_list.pop()
        return new_line_list

class yolo_obj:
    """obj based on yolo format
    """
    def __init__(self, obj_list, img_fp):
        self.class_id = int(obj_list[0])  # the label which has been changed to also reflect rounds
        self.class_id_og = int(obj_list[0])  # the actual class eg honeybee, bumblebee, unknown
        img_pil = Image.open(img_fp)
        obj_list_float = [float(x) for x in obj_list]
        self.img_w , self.img_h = img_pil.size
        self.center_px_x = obj_list_float[1] * self.img_w
        self.center_px_y = obj_list_float[2] * self.img_h
        self.bb_w = obj_list_float[3] * self.img_w
        self.bb_h = obj_list_float[4] * self.img_h
        self.min_x = self.center_px_x - (self.bb_w /2.)
        self.min_y = self.center_px_y - (self.bb_h /2.)
        self.max_x = self.center_px_x + (self.bb_w /2.)
        self.max_y = self.center_px_y + (self.bb_h /2.)
        self.is_pred_and_correct = False  # a tag for when obj is a pred and the obj has a matching hiwi/qc label

    def update_class(self, mapper):
        self.class_id = mapper.get_new_id(self.class_id_og, self.is_pred_and_correct)
        return self.class_id

    def get_iou(self, another_obj):
        my_box = torch.tensor([[self.min_x, self.min_y, self.max_x, self.max_y]])
        another_box = torch.tensor([[another_obj.min_x, another_obj.min_y, another_obj.max_x, another_obj.max_y]])
        iou = torchvision.ops.box_iou(my_box, another_box)
        return iou[0,0].item()

    def get_str(self):
        """to get the format of YOLO labels
        """
        string = f"{self.class_id} {self.center_px_x/self.img_w:.6f} {self.center_px_y/self.img_h:.6f} {self.bb_w/self.img_w:.6f} {self.bb_h/self.img_h:.6f}\n"
        return string

@click.command()
@click.option("--hiwi_labels_dir", prompt="dir of labels from previous iteration")
@click.option("--round_labels_dir", prompt="dir of predictions from current round")
@click.option("--out_dir")
@click.option("--imgs_dir", prompt="filepath of dir with images")
@click.option("--min_iou_overlap", default=0.7)
def main(hiwi_labels_dir, round_labels_dir, out_dir, imgs_dir, min_iou_overlap):
    os.makedirs(out_dir, exist_ok=True)

    """
    # copy over files from hiwis labels
    for label_fn in os.listdir(hiwi_labels_dir):
        ori_fp = os.path.join(hiwi_labels_dir, label_fn)
        new_fp = os.path.join(out_dir, label_fn)
        shutil.copy(ori_fp, new_fp)
    """

    # for each round
    # for round_i, round_labels_dir in enumerate(round_x_labels_dirs)
    round_i = 0
    if True:  # turned off rounds 
        print(f"Starting round {round_i} from dir {round_labels_dir}...")
        round_id_mapper = id_mapper(round_i + 1)

        for labels_fn in tqdm(os.listdir(round_labels_dir)):
            round_labels_fp = os.path.join(round_labels_dir, labels_fn)
            combined_labels_fp = os.path.join(out_dir, labels_fn)
            existing_labels_fp = os.path.join(hiwi_labels_dir, labels_fn)  # TODO this should be changed depending on the rounds i guess

            # handling finding the correct extension using glob
            img_glob_str = os.path.join(imgs_dir, Path(labels_fn).stem+".*")
            img_glob = glob.glob(img_glob_str)
            glob_len = len(img_glob) 
            if glob_len <= 0:
                raise Exception(f"cannot find rgb image file {img_glob_str}")
            elif glob_len > 1:
                print(f"Multiple files with same name but different extensions found:{img_glob} defaulting to first one")
            img_fn = img_glob[0]

            img_fp = os.path.join(imgs_dir, img_fn)  

            with open(existing_labels_fp, "r") as existing_labels_f:
                existing_labels_lines = labels2list(existing_labels_f)
                existing_labels_objlist = [yolo_obj(x, img_fp) for x in existing_labels_lines]

            with open(combined_labels_fp, "w") as combined_labels_f:
                with open(round_labels_fp, "r") as round_labels_f:
                    
                    round_labels_list = labels2list(round_labels_f)
                    
                    # check if its really a new object or if it was already found by the hiwi
                    round_labels_objlist = [yolo_obj(x, img_fp) for x in round_labels_list]
                    
                    labels_class_ids = np.unique(np.array([x.class_id for x in existing_labels_objlist]))
                    for class_id in labels_class_ids:
                        labels_oneclass = [x for x in existing_labels_objlist if x.class_id == class_id]
                        preds_oneclass = [x for x in round_labels_objlist if x.class_id == class_id]

                        if not preds_oneclass:
                            # no predictions for this round in the class of interest 
                            # so just move on
                            break

                        matches = find_matches(labels_oneclass, preds_oneclass, min_iou_overlap)  # also updates obj att for the matched preds

                    # update classes of all objects 
                    [x.update_class(round_id_mapper) for x in round_labels_objlist]

                    for old_obj in existing_labels_objlist:
                        obj_line_str = old_obj.get_str()
                        # write to new file 
                        combined_labels_f.write(obj_line_str)


                    for round_obj in round_labels_objlist:
                        new_obj_line_str = round_obj.get_str()

                        # write to new file 
                        combined_labels_f.write(new_obj_line_str)


if __name__ == '__main__':
    main()

