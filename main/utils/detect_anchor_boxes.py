


import argparse
from sklearn.cluster import KMeans
import numpy as np
from main.config.create_config import load_dict
from data_augmentation.data_aug.bbox_util import *

CONFIG = "squeeze.config"

def detect_anchor_boxes(gt_file =  "labels.txt"):

    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()

    bboxes_sizes = []
    count_gt_files = 0
    count_gts = 0
    classes_gt = {}

    for gt_name in gt_names:

        with open(gt_name, 'r') as f:
            lines = f.readlines()
        f.close()
        count_gt_files += 1

        # each line is an annotation bounding box
        for line in lines:
            obj = line.strip().split(' ')

            if len(obj) > 7:  # check for no label given?
                gt_class = obj[0]
                gt_class_count = 1
                if gt_class in classes_gt:
                    gt_class_count = classes_gt[gt_class] + 1
                classes_gt[gt_class] = gt_class_count

                # get coordinates
                xmin = float(obj[4])
                ymin = float(obj[5])
                xmax = float(obj[6])
                ymax = float(obj[7])
                bboxes_sizes.append([xmax - xmin, ymax - ymin])
                count_gts += 1


    print("Analysed {} labels from {} label files\n".format(count_gts , count_gt_files))

    for gt_class in classes_gt:
        print("Class {} count {} \n".format(gt_class, classes_gt[gt_class]))

    X = np.array(bboxes_sizes)
    kmeans = KMeans(n_clusters=16, random_state=0).fit(X)
    print(kmeans.cluster_centers_)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract case sub-images from originals')

    parser.add_argument("--gt", help="File with gt names. DEFAULT: labels.txt")

    args = parser.parse_args()

    gt_file = "labels.txt"

    if args.gt is not None:
        gt_file = args.gt

    detect_anchor_boxes(gt_file)


