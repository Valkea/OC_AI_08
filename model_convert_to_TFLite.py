#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import tensorflow as tf
from tensorflow import keras

import segmentation_models as sm


def tf_to_tflite(source1):

    # Load Tensorflow model
    # model = keras.models.load_model(f"{source1}.h5")
    # model.load_weights(f"{source2}.hdf5")

    model = keras.models.load_model(
        f"{source1}.keras",
        custom_objects={
            "iou_score": sm.metrics.iou_score,
            "f1-score": sm.metrics.f1_score,
            "dice_loss": sm.losses.DiceLoss(),
        },
    )

    # Convert to TF-Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Write to .tflite file
    with open(f"{source1}.tflite", "wb") as f_out:
        f_out.write(tflite_model)


if __name__ == "__main__":

    # Initialize arguments parser
    def file_choices(choices, fname):
        ext = os.path.splitext(fname)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return os.path.splitext(fname)[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keras",
        type=lambda s: file_choices(("keras"), s),
        help="The path to the .keras file",
    )
    args = parser.parse_args()

    # Convert
    print(">>> LET'S CONVERT A NEW TF-LITE MODEL")
    tf_to_tflite(args.keras)
    print(">>> NEW MODEL AVAILABLE")
