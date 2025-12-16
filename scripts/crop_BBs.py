"""script to crop patches for each bb
crops patches for each bb
size of patch must be larger than the largest bb 

"""

import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import click
import yaml
from pathlib import Path
from matplotlib import font_manager
import random


def check_isin(x, x_min, x_max):
   return x > x_min and x < x_max


@click.command()
@click.option(
    "--images_dir",
    type=str,
    help="path to the rgb directory",
    required=True,
)
@click.option(
    "--labels_dir",
    type=str,
    help="path to the labels directory. theses are labels for training",
    required=True,
)
@click.option(
    "--patching_labels_dir",
    type=str,
    help="path to the labels directory for where we should patch (it can be wrong/un-QC-ed)",
    required=False,
    default=None
)
@click.option(
    "--output_dir",
    type=str,
    help="path to the output directory",
    required=True,
)
@click.option(
    "--num_repeats",
    type=int,
    help="number of repeated random crops per bb.",
    required=True,
)
@click.option(
    "--patch_size",
    type=int,
    help="size of output image",
    required=True,
)
@click.option(
    "--num_rand_patches",
    type=int,
    help="if no --patching_labels_dir set, the number of random patches to generate per image",
    required=False,
    default=0,
)
def crop_bbs(images_dir, labels_dir, patching_labels_dir, output_dir, num_repeats, patch_size, num_rand_patches):
    """
    Function to draw bbs (from YOLO format txt labels) onto images
    Some parts were written by Julian Bauer
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    filenames = os.listdir(images_dir)

    is_random_patch = patching_labels_dir is None  # for ablations
    if is_random_patch and num_rand_patches <=0:
        raise ValueError(f"no --patching_labels_dir, please set --num_rand_patches to a natural number")

    for filename in tqdm(filenames):
        try:
            img = Image.open(os.path.join(images_dir, filename))
        except:
            print(f"Skipping {filename} because PIL cannot read file as image.")

        if not is_random_patch:
            patching_f = open(os.path.join(patching_labels_dir, Path(filename).stem + ".txt"), "r")
        # drawing_f = open(os.path.join(labels_dir, Path(filename).stem + ".txt"), "r")
        img = ImageOps.exif_transpose(
            img
        )  # for smartphone images to be oriented correctly

        w, h = img.size
        coord = np.zeros((5))

        def one_patch(new_x_min, new_y_min):
            new_x_max = new_x_min + patch_size
            new_y_max = new_y_min + patch_size

            # make sure the crop does not go out of bounds on the max side
            if new_x_max > w:
                new_x_min = w - patch_size -1
                new_x_max = new_x_min + patch_size
            if new_y_max > h:
                new_y_min = h - patch_size -1
                new_y_max = new_y_min + patch_size

            new_filename = Path(filename).stem + f"_{new_x_min}_{new_y_min}_{patch_size}" + Path(filename).suffix

            new_rect = [new_x_min, new_y_min, new_x_max, new_y_max]
            img_cropped = img.crop(new_rect)
            img_cropped.save(os.path.join(output_dir, "images", new_filename))

            # iterate over all labels to put bb in newly cropped patch
            with open(os.path.join(labels_dir, Path(filename).stem + ".txt"), "r") as old_label_f:
                with open(os.path.join(output_dir, "labels", Path(new_filename).stem + ".txt"), "w") as new_label_f:
                    for label_str in old_label_f:
                        old_l = label_str.split(" ")  # TODO write some function to parse these strings
                        x_c = w * float(old_l[1])
                        y_c = h * float(old_l[2])
                        dw = w * float(old_l[3])
                        dh = h * float(old_l[4].strip())

                        x_min = x_c - dw / 2
                        x_max = x_c + dw / 2
                        y_min = y_c - dh / 2
                        y_max = y_c + dh / 2

                        # check if the bounding box is in the cropped patch or not 
                        # assumes that the bb is smaller than the patches 
                        is_in_x = check_isin(x_min, new_x_min, new_x_max) or check_isin(x_max, new_x_min, new_x_max)
                        is_in_y = check_isin(y_min, new_y_min, new_y_max) or check_isin(y_max, new_y_min, new_y_max)
                        is_in = is_in_x and is_in_y 

                        if is_in:
                            # transform to new coords and normalise
                            new_x = (int(round(x_c)) - new_x_min) 
                            new_y = (int(round(y_c)) - new_y_min)

                            new_x /= patch_size
                            new_y /= patch_size

                            # handle OOB cases
                            new_x = min(new_x, 1.0)
                            new_y = min(new_y, 1.0)
                            new_x = max(new_x, 0.0)
                            new_y = max(new_y, 0.0)

                            # save label txt 
                            new_str = f"{old_l[0]} {new_x} {new_y} {float(old_l[3])*w/patch_size} {float(old_l[4])*h/patch_size}\n"
                            new_label_f.write(new_str)

        if is_random_patch:
            # choose random origin 
            new_x_min = random.randint(0,w-patch_size)
            new_y_min = random.randint(0,h-patch_size)
            
            one_patch(new_x_min, new_y_min)
        else:
            for i, line in enumerate(patching_f):
                l = line.split(" ")
                coord[0] = l[1]
                coord[1] = l[2]
                coord[2] = l[3]
                coord[3] = l[4][:-1]
                coord[4] = l[0]

                x_c = w * coord[0]
                y_c = h * coord[1]
                dw = w * coord[2]
                dh = h * coord[3]

                x_min = x_c - dw / 2
                x_max = x_c + dw / 2
                y_min = y_c - dh / 2
                y_max = y_c + dh / 2

                # To make sure that the bb is not cropped out during augmentation
                max_dx = int(round((patch_size - dw) / 2.))
                max_dy = int(round((patch_size - dh) / 2.))

                for n in range(num_repeats):
                    dx_random = random.randint(-max_dx, max_dx+1)
                    dy_random = random.randint(-max_dy, max_dy+1)
                    
                    # center of crop
                    new_x_c = int(x_c + dx_random)
                    new_y_c = int(y_c + dy_random)

                    # new patched image crs
                    new_x_min = int(round(new_x_c - (patch_size/2)))
                    new_y_min = int(round(new_y_c - (patch_size/2)))
                    new_x_min = max(new_x_min, 0)
                    new_y_min = max(new_y_min, 0)

                    
                    one_patch(new_x_min, new_y_min)



if __name__ == "__main__":
    seed=2378
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    crop_bbs()
