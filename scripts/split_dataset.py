#!/usr/bin/env python3
"""script to split data into train-val-test 
data dir should have either one of the following structure 

-- parent_dir
      -- AAA.jpg <RGB image>
      -- AAA.txt <exported from CVAT, in YOLO format>
      -- BBB.jpg 
      -- BBB.txt 
      ...
  -- train.txt <exported from CVAT, file containing all names of images>

if train.txt does not exist, we will try to read from the dir <parent_dir>/images
in that case, the struct of the dirs will be
-- parent_dir 
    -- images 
        -- AAA.jpg
        ...
    -- labels
        -- AAA.txt

"""


import os
import shutil
import argparse
import random

import numpy as np 
from sklearn.model_selection import train_test_split


def print_stats(count, tag):
    total = np.sum(count)
    print(f"count_{tag}: {count}, total {tag}: {total}, percentages {tag}: {count/total*100.}")

def count_instance(parent_file, num_classes):
    count=np.zeros(num_classes)
    # open file 
    f = open(parent_file, 'r')
    f_l = f.readlines()
    c_l = [int(obj[0]) for obj in f_l]
    np_c = np.unique(c_l, return_counts=True)

    for i, c in enumerate(np_c[0]):
        count[c]=np_c[1][i]

    return count



def count_instance_dir(label_dir, num_classes):
    count=np.zeros(num_classes)

    for fn in os.listdir(label_dir):
        # TODO check valid label file
        count = count + count_instance(os.path.join(label_dir, fn), num_classes)

    return count 


def write_l_to_file(fp, l):
    f = open(fp, 'w')
    f.write(''.join(l))

def move_to_dir(target_dir, source_l, parent_dir):
    target_dir_imgs = os.path.join(target_dir, "images")
    target_dir_labels = os.path.join(target_dir, "labels")
    os.mkdir(target_dir)
    os.mkdir(target_dir_imgs)
    os.mkdir(target_dir_labels)

    if os.path.exists(os.path.join(parent_dir, source_l[0])):
        img_dir = parent_dir
        labels_dir = parent_dir
    else:
        img_dir = os.path.join(parent_dir, "images")
        labels_dir = os.path.join(parent_dir, "labels")

    for fn in source_l:
        shutil.copy2(os.path.join(img_dir, fn), target_dir_imgs)
        label_fn = fn.split('.')[0] + ".txt" # also move to labels
        shutil.copy2(os.path.join(labels_dir, label_fn), target_dir_labels)


if __name__ == "__main__":

    seed=2378
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(
            prog="split_dataset",
            description="split dataset to train test val",
            epilog=""
            )

    parser.add_argument("parent_dir")
    parser.add_argument("--test_split", 
            type=float,
            default=0.15,
            help="ratio of test split to the full dataset.")

    parser.add_argument("--val_split", 
            type=float,
            default=0.15,
            help="ratio of var split to the full dataset.")


    args = parser.parse_args()

    test_split = args.test_split
    val_split = args.val_split
    parent_dir = args.parent_dir

    full_txt=os.path.join(parent_dir, "train.txt")
    num_classes = 3

    if not os.path.exists(full_txt):
        print(f"train.txt is not found. instead reading list of images from {parent_dir}/images")
        full_l = os.listdir(os.path.join(parent_dir, "images"))
    else:
        full_l = open(full_txt, 'r').read().splitlines() 

    # split the data 
    if test_split > 0:
        train_l, test_l, _, _ = train_test_split(full_l, full_l, 
                                             test_size=test_split, 
                                             )
    else:
        train_l = full_l

    train_l, val_l, _, _ = train_test_split(train_l, train_l, 
                                         test_size=int(val_split*len(full_l)), 
                                         )

    if test_split > 0:
        print(f"train size: {len(train_l)}\nval size: {len(val_l)}\ntest size: {len(test_l)}")
    else:
        print(f"train size: {len(train_l)}\nval size: {len(val_l)}\n")

    # move files to separate dirs based on lists
    train_dir =os.path.join(parent_dir, "train")
    val_dir = os.path.join(parent_dir, "val")
    if test_split > 0:
        test_dir = os.path.join(parent_dir, "test")

    move_to_dir(train_dir, train_l, parent_dir)
    move_to_dir(val_dir, val_l, parent_dir)
    if test_split > 0:
        move_to_dir(test_dir, test_l, parent_dir)

    count_train=count_instance_dir(os.path.join(train_dir,"labels"), 3)
    count_val=count_instance_dir(os.path.join(val_dir,"labels"), 3)
    print_stats(count_train, "train")
    print_stats(count_val, "val")
    if test_split > 0:
        count_test=count_instance_dir(os.path.join(test_dir,"labels"), 3)
        print_stats(count_test, "test")

