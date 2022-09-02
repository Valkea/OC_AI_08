#! /usr/bin/env python3
# coding: utf-8

import os
import io

# import joblib
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify, Response

import pathlib
# import html

import numpy as np
import segmentation_models as sm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# import tensorflow as tf
from tensorflow import keras

from PIL import Image


# --- Preprocessing ---

BACKBONE = "efficientnetb7"
preprocess_input = sm.get_preprocessing(BACKBONE)


def preprocess_sample(img, preprocessing=None):
    x = np.array(img)
    if preprocessing:
        x = preprocessing(x)

    return np.array([x / 255], dtype=float)


# --- Load TF Model ---

print("Load Semantic-segmentation Model")
model_name = "FPN-efficientnetb7_with_data_augmentation_2_diceLoss"
model = keras.models.load_model(
    f"models/{model_name}.keras",
    custom_objects={
        "iou_score": sm.metrics.iou_score,
        "f1-score": sm.metrics.f1_score,
        "dice_loss": sm.losses.DiceLoss(),
    },
)


# --- Uploading samples ---
UPLOAD_FOLDER = "/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "super secret key"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        if file and allowed_file(file.filename):
            print(os.getcwd())
            # filename = secure_filename(file.filename)
            image_bytes = Image.open(io.BytesIO(file.read()))

            # Preprocess image
            img = preprocess_sample(image_bytes, preprocess_input)

            # Apply model
            print("--- Predict")
            pred = model.predict(img)

            # Convert to categories
            mask = np.argmax(pred, axis=3)[0]

            # Return the matrix
            return jsonify(mask.tolist())

    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/")
def index():
    return "The 'CityScape Semantic-segmentation API' server is up."


def get_ids(path):
    ids = []
    for x in path.glob("*_labels.png"):
        path = str(x)
        file = path[path.rfind('/')+1:-11]
        ids.append(file)
    return ids


@app.route("/list/")
def file_list():
    files_path = pathlib.Path('data', 'preprocessed', "256x128", "val")
    ids = get_ids(files_path)
    table = "".join([f"<p><a href='{url_for('display', pic_id=x)}'>{x}</a></p>" for x in ids])
    return f"""
    <!doctype html>
    <title>List of ids</title>
    <h1>List of available ids</h1>
    {table}
    """

def compare_segmentations(img_source, mask_source, predictions):

    mask = np.argmax(predictions, axis=3)[0]

    fig = plt.figure(figsize=(19,10))
    plt.subplot(1,3,1)
    plt.imshow(img_source)
    plt.axis('off')
    plt.title("Source")

    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title("Predicted mask")

    plt.subplot(1,3,3)
    plt.imshow(mask_source)
    plt.axis('off')
    plt.title("Original mask")

    plt.tight_layout()
    plt.show()
    return fig

@app.route("/display/<pic_id>", methods=["GET", "POST"])
def display(pic_id):

    test_img = Image.open(str(pathlib.Path('data','preprocessed', '256x128', 'val', f"{pic_id}.png")))
    test_mask = Image.open(str(pathlib.Path('data','preprocessed', '256x128', 'val', f"{pic_id}_labels.png")))
    preprocessed_img = preprocess_sample(test_img, preprocess_input)

    predict = model.predict(np.array(preprocessed_img))
    compare_segmentations(test_img, test_mask, predict)

    fig = compare_segmentations()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

    return f"""
    <!doctype html>
    <title>Predict display</title>
    <h1>Display result</h1>
    {pic_id}
    """


if __name__ == "__main__":
    current_port = int(os.environ.get("PORT") or 5000)
    app.run(debug=True, host="0.0.0.0", port=current_port)
