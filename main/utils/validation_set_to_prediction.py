

import argparse
import cv2
import os
from main.config.create_config import load_dict
from main.utils.image_standardizer import standardize_image_size, preprocess_image
from shutil import copy

CONFIG = "squeeze.config"
OUTPUT_DIR = "./images-to-predict/"

def validation_set_to_prediction(val_img_file):


    with open(val_img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()

    for count in range(0,len(img_names)):
        img_name = img_names[count]

        copy(img_name, OUTPUT_DIR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Copies validation set images to prediction dir')

    parser.add_argument("--img", help="File with raw image names. DEFAULT: img_val.txt")

    args = parser.parse_args()

    val_img_file = "img_val.txt"

    if args.img is not None:
        img_file = args.img

    validation_set_to_prediction(val_img_file)


