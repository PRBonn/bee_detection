"""randomly select patches incrementall
"""

import os
import random
import shutil
import click

import sys
from split_dataset import count_instance_dir



@click.command()
@click.option(
    "--parent_dir",
    type=str,
    help="path to the data folder containing the combination of training and val patched images",
)
@click.option(
    "--output_parent_dir",
    type=str,
    help="path to the output folder",
)
def main(parent_dir, output_parent_dir):
    
    imgs_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "labels")

    imgs_names = os.listdir(imgs_dir)
    imgs_names.sort()
    random.shuffle(imgs_names)

    for size in range(100,1000,100):
        output_dir = os.path.join(output_parent_dir, f"train_{size}")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))
        os.makedirs(os.path.join(output_dir, "labels"))
        for img_name in imgs_names:
            shutil.copy2(
                    os.path.join(imgs_dir, img_name),
                    os.path.join(output_dir, "images", img_name),
                    )
            shutil.copy2(
                    os.path.join(labels_dir, img_name.split(".")[0]+".txt"),
                    os.path.join(output_dir, "labels", img_name.split(".")[0]+".txt"),
                    )
            num_bees = count_instance_dir(os.path.join(output_dir, 'labels'), 3).sum()
            if num_bees >= size:
                break


    val_size = 100
    val_list = imgs_names[-val_size:]

    output_dir = os.path.join(output_parent_dir, f"val_{val_size}")           
    os.makedirs(output_dir)                                                 
    os.makedirs(os.path.join(output_dir, "images"))                         
    os.makedirs(os.path.join(output_dir, "labels")) 
    for img_name in val_list:
        shutil.copy2(
               os.path.join(imgs_dir, img_name),
               os.path.join(output_dir, "images", img_name),
               )
        shutil.copy2(
               os.path.join(labels_dir, img_name.split(".")[0]+".txt"),
               os.path.join(output_dir, "labels", img_name.split(".")[0]+".txt"),
               )


if __name__ == '__main__':
    random.seed(0)
    main()
