


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
from pathlib import Path
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import cv2


log_dir_name = "./log"
checkpoint_dir = './log/checkpoints'
CUDA_VISIBLE_DEVICES = "1"
steps = None
GPUS = 1
CONFIG = "squeeze.config"
K.set_learning_phase(0)

def export_full_model():

    #create config object
    cfg = load_dict(CONFIG)

    prepare_and_export_model(cfg)


def prepare_and_export_model(cfg):

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

    exportName = checkpoint_dir + "/" + ckpt + ".full"
    model.save(exportName)

    frozen_graph_path = checkpoint_dir + u"\\model.pb"
    to_frozen_graph(model, frozen_graph_path)
    to_tflite_from_frozen_graph(frozen_graph_path, checkpoint_dir)

def to_tflite_from_frozen_graph(frozen_graph_path, output_file_dir):
    inputs = ["input"]
    outputs = ["reshape_1/Reshape"]

    converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_graph_path, inputs, outputs)
    tflite_model = converter.convert()
    save_model(output_file_dir + u"\\model.tflite", tflite_model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    save_model(output_file_dir + u"\\model_size_optimized.tflite", tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    tflite_model = converter.convert()
    save_model(output_file_dir + u"\\model_float16.tflite", tflite_model)


def save_model(file_path, model):
    with open(file_path, "wb") as f:
        f.write(model)

def to_frozen_graph(model, output_model_path):
    output_fld = Path(output_model_path).parent
    output_model_name = Path(output_model_path).name
    output_model_stem = Path(output_model_path).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # K.set_image_data_format('channels_first')
    K.set_image_data_format('channels_last')

    output_node_names = [node.op.name for node in model.outputs]

    sess = K.get_session()
    tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                         output_model_pbtxt_name, as_text=True)

    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_node_names
    )

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name, as_text=False)

if __name__ == "__main__":

    #argument parsing
    parser = argparse.ArgumentParser(description='Exports full model with structure and weights.')
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--steps",  type=int, help="steps to evaluate. DEFAULT: length of imgs/ batch_size")
    parser.add_argument("--gpu",  help="gpu to use. DEFAULT: 1")
    parser.add_argument("--gpus",  type=int, help="gpus to use for multigpu usage. DEFAULT: 1")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")

    args = parser.parse_args()

    #set global variables according to optional arguments
    if args.logdir is not None:
        log_dir_name = args.logdir
        checkpoint_dir = log_dir_name + '/checkpoints'

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


    export_full_model()