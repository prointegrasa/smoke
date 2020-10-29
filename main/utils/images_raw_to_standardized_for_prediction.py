

import argparse
import cv2
import os
from main.config.create_config import load_dict
from main.utils.image_standardizer import standardize_image_size, preprocess_image
from shutil import copyfile

CONFIG = "squeeze.config"
OUTPUT_DIR = "./images-to-predict-standardized/"

def standardize_raw_images_for_prediction(img_file):


    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()

    #create config object
    cfg = load_dict(CONFIG)

    for count in range(0,len(img_names)):
        img_name = img_names[count]

        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

        img = standardize_image_size(img, cfg.IMAGE_WIDTH_PROCESSING, cfg.IMAGE_HEIGHT_PROCESSING)
        img = preprocess_image(img)

        standardized_img_name = OUTPUT_DIR + os.path.basename(img_name)

        cv2.imwrite(standardized_img_name, img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Standardizes raw images for prediction')

    parser.add_argument("--img", help="File with raw image names. DEFAULT: images_to_predict.txt")

    args = parser.parse_args()

    img_file = "images_to_predict.txt"

    if args.img is not None:
        img_file = args.img

    standardize_raw_images_for_prediction(img_file)


