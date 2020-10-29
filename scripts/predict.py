


from main.model.squeezeDet import  SqueezeDet
from main.model.dataGenerator import generator_from_data_path, visualization_generator_from_data_path, read_images_raw, prepare_images_raw_for_predict

import keras.backend as K
from keras import optimizers
import tensorflow as tf
from main.model.evaluation import simple_prediction, generate_TTA_transforms, update_best_results_by_TTA
import main.utils.utils as utils

import argparse
from keras.utils import multi_gpu_model
from main.config.create_config import load_dict
from data_augmentation.data_aug.data_aug import *

import time
import cv2

#default values for some variables
img_file = "images_to_predict.txt"

log_dir_name = "./log"
checkpoint_dir = './log/checkpoints'
CUDA_VISIBLE_DEVICES = "1"
steps = None
GPUS = 1
CONFIG = "squeeze.config"
PREDICTION_POSTFIX = ".prediction"
RESULTS_DIR = "./prediction-results/"


def predict():

    #create config object
    cfg = load_dict(CONFIG)

    #if multigpu support, adjust batch size
    if GPUS > 1:
        cfg.BATCH_SIZE = GPUS * cfg.BATCH_SIZE

    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()

    nbatches, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    model = prepare_model(cfg)

    all_results = {}

    t0 = time.clock()

    for i in range(0,nbatches):
        img_names_batch = img_names[i*cfg.BATCH_SIZE: (i+1)*cfg.BATCH_SIZE]
        all_results = predict_on_files(cfg, model, img_names_batch, all_results)

    if mod > 0:
        img_names_batch = img_names[-mod:]
        all_results = predict_on_files(cfg, model, img_names_batch, all_results)

    t1 = time.clock()
    print("Total prediction time (sec): ", t1 - t0)  # CPU seconds elapsed (floating point)

    if cfg.USE_PRED_FINAL_PRESENTATION:
        results_prefix = "Linear scale: "
    else:
        results_prefix = "Raw class: "

    for key in all_results:
        print(results_prefix + str(key) + " objects: " + str(all_results[key]))


def prepare_model(cfg):

    #set gpu to use if no multigpu

    #hide the other gpus so tensorflow only uses this one
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)


    #instantiate model
    squeeze = SqueezeDet(cfg)

    #dummy optimizer for compilation
    sgd = optimizers.SGD(lr=cfg.LEARNING_RATE, decay=0, momentum=cfg.MOMENTUM,
                         nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)




    if GPUS > 1:

        #parallelize model
        model = multi_gpu_model(squeeze.model, gpus=GPUS)
        model.compile(optimizer=sgd,
                              loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                            squeeze.conf_loss, squeeze.loss_without_regularization])


    else:
    #compile model from squeeze object, loss is not a function of model directly
        squeeze.model.compile(optimizer=sgd,
                              loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                            squeeze.conf_loss, squeeze.loss_without_regularization])

        model = squeeze.model

    # load first model checkpoint available in log dir
    ckpt = sorted(os.listdir(checkpoint_dir))[0]
    model.load_weights(checkpoint_dir + "/" + ckpt)
    return model

def create_final_pred_presentation(image_area, object_area, object_class):

    area_ratio = object_area / image_area
    text = "Danger:"
#linear scale of six values

    if object_class == 0:
        linear_result = 1
        if area_ratio < 0.015:
            linear_result -= 1
        elif area_ratio > 0.04:
            linear_result += 1
    else:
        linear_result = 4
        if area_ratio < 0.035:
            linear_result -= 1
        elif area_ratio > 0.095:
            linear_result += 1



    if linear_result == 0:
        text += " Very Low"
        color = (0, 255, 0)
    elif linear_result == 1:
        text += " Low"
        color = (0, 255, 0)
    elif linear_result == 2:
        text += " Medium"
        color = (0, 128, 255)
    elif linear_result == 3:
        text += " High"
        color = (0, 128, 255)
    elif linear_result == 4:
        text += " Very High"
        color = (0, 0, 255)
    else:
        text += "Extreme"
        color = (0, 0, 255)


    return text, color, linear_result

def predict_on_files(cfg, model, img_names, all_results):

    #compute number of batches per epoch
    nbatches_valid, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    #if a number for steps was given
    if steps is not None:
        nbatches_valid = steps

    images_to_predict_raw = read_images_raw(img_names)
    images_to_predict = prepare_images_raw_for_predict(images_to_predict_raw, cfg, None)
    batches = []
    batches.append(images_to_predict)

    # predict on original images
    all_classes, all_scores, all_boxes = simple_prediction(model=model, generator=batches, steps=nbatches_valid,
                                                           config=cfg)
    # we have only one batch
    best_classes =  all_classes[0]
    best_scores =  all_scores[0]
    best_boxes =  all_boxes[0]

    tta_updated_indexes = set()

    if cfg.USE_TTA_ON_PREDICT == 1:

        tta_transforms, tta_reverse_transforms = generate_TTA_transforms()

        # predict on transformed copies to get best possible result

        for tta_transform_index in range(0,len(tta_transforms)):

            images_to_predict_raw_by_transform = images_to_predict_raw.copy()

            for image_count in range(0, len(images_to_predict_raw_by_transform)):
                transformed_image_raw, bbox = tta_transforms[tta_transform_index](images_to_predict_raw_by_transform[image_count], None)
                images_to_predict_raw_by_transform[image_count] = transformed_image_raw

            images_to_predict = prepare_images_raw_for_predict(images_to_predict_raw_by_transform, cfg, None)
            batches = []
            batches.append(images_to_predict)

            all_classes, all_scores, all_boxes = simple_prediction(model=model, generator=batches, steps=nbatches_valid,
                                                                   config=cfg)

            best_classes, best_scores, best_boxes, tta_updated_indexes = \
                update_best_results_by_TTA(images_to_predict, tta_reverse_transforms[tta_transform_index],
                best_classes, best_scores, best_boxes,
                all_classes[0], all_scores[0], all_boxes[0], tta_updated_indexes,
                cfg.USE_TTA_MAXIMIZE_PRECISION == 1)

    # graphical presentation and reporting
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    image_area = cfg.IMAGE_WIDTH_PROCESSING * cfg.IMAGE_HEIGHT_PROCESSING

    for img_name in img_names:
        report_text_file = RESULTS_DIR + os.path.basename(img_name) + PREDICTION_POSTFIX + ".txt"
        report_image_file = RESULTS_DIR + os.path.basename(img_name) + PREDICTION_POSTFIX + ".png"

        original_img = cv2.imread(img_name)

        for count_in_image in range(0, len(best_boxes[count])):

            bbox = utils.bbox_transform_single_box(best_boxes[count][count_in_image])
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            #write graphic report (original images with prediction results)

            if cfg.USE_PRED_FINAL_PRESENTATION:
                object_area = (xmax - xmin) * (ymax - ymin)
                text, color, linear_result = create_final_pred_presentation(image_area, object_area, best_classes[count][count_in_image])

                results = 0
                if linear_result in all_results:
                    results = all_results[linear_result]
                results += 1
                all_results[linear_result] = results

                # add rectangle and text
                cv2.rectangle(original_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
                cv2.putText(original_img,text,
                            (int(xmin), int(ymin)), font, 0.5, color, 1, cv2.LINE_AA)
            else:
                # add rectangle and text
                cv2.rectangle(original_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
                class_found = cfg.CLASS_NAMES[best_classes[count][count_in_image]]
                text = class_found +":" + str(round(best_scores[count][count_in_image],2))

                class_results = 0
                if class_found in all_results:
                    class_results = all_results[class_found]
                class_results += 1
                all_results[class_found] = class_results

                if cfg.USE_TTA_ON_PREDICT == 1:
                    if count in tta_updated_indexes:
                        text += "TTA"

                cv2.putText(original_img,text,
                            (int(xmin), int(ymin)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite(report_image_file, original_img)

        #write text report

        with open(report_text_file, 'w') as f:

            rows = []

            for count_in_image in range(0, len(best_boxes[count])):

                bbox = utils.bbox_transform_single_box(best_boxes[count][count_in_image])
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]

                labelrow = cfg.CLASS_NAMES[best_classes[count][count_in_image]] + " " + str(best_scores[count][count_in_image]) + " " +\
                               str(xmin) + " " + \
                               str(ymin) + " " + \
                               str(xmax) + " " + \
                               str(ymax) + "\n"

                rows.append(labelrow)

            f.writelines(rows)
        f.close()
        count += 1

    return all_results


if __name__ == "__main__":

    #argument parsing
    parser = argparse.ArgumentParser(description='Evaluate squeezeDet keras checkpoints after each epoch on validation set.')
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--test_img", help="file of full path names for the test images. DEFAULT: images_to_predict.txt")
    parser.add_argument("--steps",  type=int, help="steps to evaluate. DEFAULT: length of imgs/ batch_size")
    parser.add_argument("--gpu",  help="gpu to use. DEFAULT: 1")
    parser.add_argument("--gpus",  type=int, help="gpus to use for multigpu usage. DEFAULT: 1")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")

    args = parser.parse_args()

    #set global variables according to optional arguments
    if args.logdir is not None:
        log_dir_name = args.logdir
        checkpoint_dir = log_dir_name + '/checkpoints'

    if args.test_img is not None:
        img_file_test = args.test_img

    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu


    if args.gpus is not None:
        GPUS = args.gpus


        #if there were no GPUS explicitly given, take the last ones
        #the assumption is, that we use as many gpus for evaluation as for training
        #so we have to hide the other gpus to not try to allocate memory there
        if args.gpu is None:
            CUDA_VISIBLE_DEVICES = ""
            for i in range(GPUS, 2*GPUS):
                CUDA_VISIBLE_DEVICES += str(i) + ","
            print(CUDA_VISIBLE_DEVICES)

    if args.steps is not None:
        steps = args.steps

    if args.config is not None:
        CONFIG = args.config


    predict()