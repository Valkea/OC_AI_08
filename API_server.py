#! /usr/bin/env python3
# coding: utf-8

import os
import io
import pathlib

from flask import Flask, flash, request, redirect, jsonify

import numpy as np
from PIL import Image

# import tflite_runtime.interpreter as tflite

import onnxruntime as rt
print("ONX:", rt.get_device())


# ########## API ##########


# --- Load TF Model ---

base_W = 512
base_H = 256
base_resolution = f"{base_W}x{base_H}"

print("Load Semantic-segmentation Model")
model_name = "FPN-efficientnetb7_with_data_augmentation_2_diceLoss_512x256"

# -- with a keras model
# model = keras.models.load_model(
#     f"models/{model_name}.keras",
#     custom_objects={
#         "iou_score": sm.metrics.iou_score,
#         "f1-score": sm.metrics.f1_score,
#         "dice_loss": sm.losses.DiceLoss(),
#     },
# )

# -- with a TF-Lite model
# interpreter = tflite.Interpreter(model_path=f"models/{model_name}.tflite")
# interpreter.resize_tensor_input(0, [1, base_H, base_W, 3])
# interpreter.allocate_tensors()
# input_index = interpreter.get_input_details()[0]["index"]
# output_index = interpreter.get_output_details()[0]["index"]

# --- with a ONNX model

# providers = ['CPUExecutionProvider']
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
m = rt.InferenceSession(str(pathlib.Path('models', f"{model_name}.onnx")), providers=providers)


# --- API Flask app ---

app = Flask(__name__)
app.secret_key = "super secret key"


UPLOAD_FOLDER = "/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return "The 'CityScape Semantic-segmentation API' server is up."


@app.route("/predict/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and (allowed_file(file.filename) or file.filename == 'file'):
            print(os.getcwd())
            # filename = secure_filename(file.filename)
            image_bytes = Image.open(io.BytesIO(file.read()))

            # Preprocess image
            # img = preprocess_sample(image_bytes, preprocess_input)
            # /!\ Preprocessed layers are now included in the model
            img = np.array([np.array(image_bytes)], dtype=np.float32)

            if (img.shape[1] != base_H or img.shape[2] != base_W):
                raise Exception(f"Custom Error: wrong image size ({base_H}x{base_W}) required!")

            # Apply model
            print("--- Predict")
            # pred = model.predict(img)  # keras model

            img = np.array(img, dtype=np.float32)
            print(img.shape)

            # -- Predict with TF-Lite
            # interpreter.set_tensor(input_index, img)
            # interpreter.invoke()
            # pred = interpreter.get_tensor(output_index)

            # -- Predict with ONNX
            pred = m.run(['model_6'], {'input': img})[0]

            # Convert to categories
            mask = np.argmax(pred, axis=3)[0]

            # Return the matrix
            return jsonify(mask.tolist())

    return """
    <!doctype html>
    <html>
        <head>
            <title>Upload new File</title>
        </head>
        <body>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
        </body>
    </html>
    """

# ########## START API ##########


if __name__ == "__main__":
    current_port = int(os.environ.get("PORT") or 5000)
    app.run(debug=True, host="0.0.0.0", port=current_port)
