

import argparse
import cv2
import os
from main.config.create_config import load_dict
from main.utils.image_standardizer import standardize_image_size
from shutil import copyfile

CONFIG = "squeeze.config"
OUTPUT_IMAGE_DIR = "./images/"
OUTPUT_LABEL_DIR = "./labels/"
STANDARDIZED_IMAGE_NAME = "image"

def standardize_raw_images(img_file):


    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()

    #create config object
    cfg = load_dict(CONFIG)
    image_serie = cfg.IMAGE_SERIE

    for count in range(0,len(img_names)):
        img_name = img_names[count]

        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

        img = standardize_image_size(img, cfg.IMAGE_WIDTH_STORAGE, cfg.IMAGE_HEIGHT_STORAGE)

        cv2.imwrite(OUTPUT_IMAGE_DIR + STANDARDIZED_IMAGE_NAME + "-serie-" + str(image_serie) + "-" + str(count) + ".png", img)

        #create empty labels file

        with open(OUTPUT_LABEL_DIR + STANDARDIZED_IMAGE_NAME + "-serie-" + str(image_serie) + "-" + str(count) + ".txt", 'w') as f:
            f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Standardizes raw image sizes and names')

    parser.add_argument("--img", help="File with raw image names. DEFAULT: original_images_raw.txt")

    args = parser.parse_args()

    img_file = "original_images_raw.txt"

    if args.img is not None:
        img_file = args.img

    standardize_raw_images(img_file)


