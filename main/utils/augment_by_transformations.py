


import argparse
import json
from main.config.create_config import load_dict
from data_augmentation.data_aug.data_aug import *
from data_augmentation.data_aug.bbox_util import *
from easydict import EasyDict as edict
from main.utils.image_standardizer import preprocess_image


AUGMENTED_FILE_POSTFIX = "_copy_"
AUG_CONFIG_DIR = "./images-aug-configs"
CONFIG = "squeeze.config"
AUGMENTATION_COUNT = 1

def augment_image_by_scale_and_horizontal_flip(img, img_name, bboxes_arr, classes, gt_name, IMAGE_WIDTH, IMAGE_HEIGHT, scale):

    global AUGMENTATION_COUNT
    augmenteg_img_name = os.path.splitext(img_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".png"
    augmenteg_labels_name = os.path.splitext(gt_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".txt"
    AUGMENTATION_COUNT += 1
    skipped_samples = 0

    bboxes = np.array(bboxes_arr)
    transforms = Sequence([Scale(scale_x=scale, scale_y=scale)])
    new_img, new_bboxes = transforms(img, bboxes)

    skip_augmented_sample = False

    if len(bboxes) > 0 and len(new_bboxes) != len(bboxes):
        skip_augmented_sample = True

    for count in range(0, len(new_bboxes)):
        xmin = new_bboxes[count][0]
        ymin = new_bboxes[count][1]
        xmax = new_bboxes[count][2]
        ymax = new_bboxes[count][3]
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            skip_augmented_sample = True
        if xmin > IMAGE_WIDTH or xmax > IMAGE_WIDTH:
            skip_augmented_sample = True
        if ymin > IMAGE_HEIGHT or ymax > IMAGE_HEIGHT:
            skip_augmented_sample = True

    if skip_augmented_sample:
        print("Skipped augmentation " + os.path.basename(img_name) + " scale " + str(scale))
        skipped_samples += 1

    if not skip_augmented_sample:
        cv2.imwrite(augmenteg_img_name, new_img)

        with open(augmenteg_labels_name, 'w') as f:

            for count in range(0, len(new_bboxes)):
                class_name = classes[count]
                xmin = new_bboxes[count][0]
                ymin = new_bboxes[count][1]
                xmax = new_bboxes[count][2]
                ymax = new_bboxes[count][3]

                labelrow = class_name + " 0 0 0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                    ymax) + "  0 0 0 0 0 0 0\n"

                f.write(labelrow)
        f.close()

    ###

    augmenteg_img_name = os.path.splitext(img_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".png"
    augmenteg_labels_name = os.path.splitext(gt_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".txt"
    AUGMENTATION_COUNT += 1
    bboxes = np.array(bboxes_arr)

    transforms = Sequence([Scale(scale_x=scale, scale_y=scale), HorizontalFlip()])
    new_img, new_bboxes = transforms(img, bboxes)

    skip_augmented_sample = False

    if len(bboxes) > 0 and len(new_bboxes) != len(bboxes):
        skip_augmented_sample = True

    for count in range(0, len(new_bboxes)):
        xmin = new_bboxes[count][0]
        ymin = new_bboxes[count][1]
        xmax = new_bboxes[count][2]
        ymax = new_bboxes[count][3]
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            skip_augmented_sample = True
        if xmin > IMAGE_WIDTH or xmax > IMAGE_WIDTH:
            skip_augmented_sample = True
        if ymin > IMAGE_HEIGHT or ymax > IMAGE_HEIGHT:
            skip_augmented_sample = True

    if skip_augmented_sample:
        print("Skipped augmentation " + os.path.basename(img_name) + " flip+scale " + str(scale))
        skipped_samples += 1

    if not skip_augmented_sample:
        cv2.imwrite(augmenteg_img_name, new_img)

        with open(augmenteg_labels_name, 'w') as f:

            for count in range(0, len(new_bboxes)):
                class_name = classes[count]
                xmin = new_bboxes[count][0]
                ymin = new_bboxes[count][1]
                xmax = new_bboxes[count][2]
                ymax = new_bboxes[count][3]

                labelrow = class_name + " 0 0 0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                    ymax) + "  0 0 0 0 0 0 0\n"

                f.write(labelrow)
        f.close()

    return skipped_samples


def augment_image_by_transformations(img_name, gt_name, config):

    global AUGMENTATION_COUNT
    AUGMENTATION_COUNT = 1

    with open(gt_name, 'r') as f:
        lines = f.readlines()
    f.close()

    bboxes_arr = []
    classes = []

    scaleX = float(config.IMAGE_WIDTH_PROCESSING) / float(config.IMAGE_WIDTH_STORAGE)
    scaleY = float(config.IMAGE_HEIGHT_PROCESSING) / float(config.IMAGE_HEIGHT_STORAGE)

    # each line is an annotation bounding box
    for line in lines:
        obj = line.strip().split(' ')

        if len(obj) > 7: #check for no label given?
            class_name = obj[0]
            # get coordinates
            xmin = float(obj[4]) * scaleX
            ymin = float(obj[5]) * scaleY
            xmax = float(obj[6]) * scaleX
            ymax = float(obj[7]) * scaleY

            bboxes_arr.append([xmin, ymin, xmax, ymax])
            classes.append(class_name)


    img = cv2.imread(img_name)
    img = preprocess_image(img)

    # convert from storage format to processing format for train/eval/predict
    img = cv2.resize(img, (config.IMAGE_WIDTH_PROCESSING, config.IMAGE_HEIGHT_PROCESSING))
    cv2.imwrite(img_name, img)

    with open(gt_name, 'w') as f:
        for count in range(0, len(bboxes_arr)):
            class_name = classes[count]
            xmin = bboxes_arr[count][0]
            ymin = bboxes_arr[count][1]
            xmax = bboxes_arr[count][2]
            ymax = bboxes_arr[count][3]

            labelrow = class_name + " 0 0 0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                ymax) + "  0 0 0 0 0 0 0\n"

            f.write(labelrow)
    f.close()


    '''
    scaleX = float(config.IMAGE_WIDTH_PROCESSING)/float(config.IMAGE_WIDTH_STORAGE) - 1.0
    scaleY = float(config.IMAGE_HEIGHT_PROCESSING)/float(config.IMAGE_HEIGHT_STORAGE) - 1.0

    transforms = Sequence([Scale(scale_x=scaleX, scale_y=scaleY)])
    img, bboxes_arr = transforms(img, np.array(bboxes_arr))

    cv2.imwrite(img_name, img)
    with open(gt_name, 'w') as f:
        for count in range(0, len(bboxes_arr)):
            class_name = classes[count]
            xmin = bboxes_arr[count][0]
            ymin = bboxes_arr[count][1]
            xmax = bboxes_arr[count][2]
            ymax = bboxes_arr[count][3]

            labelrow = class_name + " 0 0 0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                ymax) + "  0 0 0 0 0 0 0\n"

            f.write(labelrow)
    f.close()

    '''


######
# always apply horizontal flip augmentation

    augmenteg_img_name = os.path.splitext(img_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".png"
    augmenteg_labels_name = os.path.splitext(gt_name)[0] + AUGMENTED_FILE_POSTFIX + str(AUGMENTATION_COUNT) + ".txt"
    AUGMENTATION_COUNT += 1

    bboxes = np.array(bboxes_arr)

    transforms = Sequence([HorizontalFlip()])

    new_img, new_bboxes = transforms(img, bboxes)

    cv2.imwrite(augmenteg_img_name, new_img)

    with open(augmenteg_labels_name, 'w') as f:

        for count in range(0, len(new_bboxes)):

            class_name = classes[count]
            xmin = new_bboxes[count][0]
            ymin = new_bboxes[count][1]
            xmax = new_bboxes[count][2]
            ymax = new_bboxes[count][3]

            labelrow = class_name + " 0 0 0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                ymax) + "  0 0 0 0 0 0 0\n"

            f.write(labelrow)
    f.close()

###

# try to load individual settings per image sample if exists
    aug_config_name =  AUG_CONFIG_DIR + "/" + os.path.splitext(os.path.basename(img_name))[0] + ".config"

    try:
        with open(aug_config_name, "r") as f:
            aug_cfg = json.load(f)  ### this loads the array from .json format
            aug_cfg = edict(aug_cfg)
            augmentation_scales = aug_cfg.SCALES
    except:
        # not found - default augmentation scales
        augmentation_scales = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# run augmentation by scales
    skipped_samples = 0
    for scale in augmentation_scales:
        skipped_samples += augment_image_by_scale_and_horizontal_flip(img, img_name, bboxes_arr, classes, gt_name,
                                                                      config.IMAGE_WIDTH_PROCESSING, config.IMAGE_HEIGHT_PROCESSING, scale)

    return skipped_samples


def augment_by_transformations(img_file =  "images.txt" ,
                    gt_file =  "labels.txt"):


    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()

    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()

    if len(img_names) != len(gt_names):
        raise ValueError('images and labels not of the same length')

    #create config object
    cfg = load_dict(CONFIG)

    skipped_samples = 0
    for count in range(0,len(img_names)):
        img_name = img_names[count]
        gt_name = gt_names[count]

        skipped_samples += augment_image_by_transformations(img_name, gt_name, cfg)

    print("Total samples skipped: " + str(skipped_samples))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract case sub-images from originals')

    parser.add_argument("--img", help="File with image names. DEFAULT: images.txt")
    parser.add_argument("--gt", help="File with gt names. DEFAULT: labels.txt")

    args = parser.parse_args()

    img_file = "images.txt"
    gt_file = "labels.txt"


    if args.img is not None:
        img_file = args.img

    if args.gt is not None:
        gt_file = args.gt

    augment_by_transformations(img_file, gt_file)


