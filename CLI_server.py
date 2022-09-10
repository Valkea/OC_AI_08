#! /usr/bin/env python3
# coding: utf-8

import os
import io
import pathlib
import requests

from flask import Flask, request, url_for, Response

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


# --- API Flask app ---

app = Flask(__name__)
app.secret_key = "super secret key"

base_W = 512
base_H = 256
base_resolution = f"{base_W}x{base_H}"


# ########## DEMO FRONTEND ##########
# This could be a different Flask script totally independant from the API!

# API_URL = "http://0.0.0.0:5000"
API_URL = "http://cityscape-segmentation.herokuapp.com"


@app.route("/")
def index():
    return "The 'CityScape Semantic-segmentation CLIENT' is up."


def get_ids(path):
    ids = []
    for x in path.glob("*_labels.png"):
        path = str(x)
        file = path[path.rfind('/')+1:-11]
        ids.append(file)
    return ids


@app.route("/list/")
def file_list():
    files_path = pathlib.Path('data', 'preprocessed', base_resolution, "val")
    ids = get_ids(files_path)

    fileslist = "".join([f"<p><a href='{url_for('display', pic_id=x)}'>{x}</a></p>" for x in ids])

    global API_URL
    if API_URL is None:
        API_URL = request.url_root
    if API_URL is None:
        API_URL = request.url_root

    return f"""
    <!doctype html>
    <html>
        <head>
            <title>List of available ids</title>
        </head>
        <body>
            <h1>Upload new File</h1>
            <form action={API_URL}/predict/ method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
            <h1>List of available ids</h1>
            {fileslist}
        </body>
    </html>
    """


def compare_segmentations(img_source, mask_source, mask, iou, dice):

    # mask = np.argmax(predictions, axis=3)[0]

    fig = plt.figure(figsize=(19, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_source)
    plt.axis('off')
    plt.title("Source")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title(f"Predicted mask (IoU={iou:.4f} | Dice:{dice:.4f})")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_source)
    plt.axis('off')
    plt.title("Expected mask")

    plt.tight_layout()
    # plt.show()
    return fig


def mIOU(label, pred, num_classes=8):

    iou_list = list()
    present_iou_list = list()

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


@app.route("/display/<pic_id>", methods=["GET", "POST"])
def display(pic_id):

    # load base & mask images from the given 'pic_id'
    source_img = Image.open(str(pathlib.Path('data', 'preprocessed', base_resolution, 'val', f"{pic_id}.png")))
    source_mask = Image.open(str(pathlib.Path('data', 'preprocessed', base_resolution, 'val', f"{pic_id}_labels.png")))

    # convert img to bytes for POST action
    img_byte_arr = io.BytesIO()
    source_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # POST the source_img (as a byte array)
    global API_URL
    if API_URL is None:
        API_URL = request.url_root
    files = {'file': img_byte_arr}
    res = requests.post(f'{API_URL}/predict/', files=files)

    # process API response
    predict = np.array(res.json())
    print("CLIENT: response type:", type(predict))
    print("CLIENT: responde shape:", predict.shape)

    # display the 3 images side by side for comparison (with scores)
    # y_true = np.eye(8)[source_mask]
    # y_pred = np.eye(8)[predict]
    y_true2 = np.asarray(source_mask)
    y_pred2 = predict
    iou = float(mIOU(y_true2, y_pred2, 8))
    dice = (2*iou)/(iou+1)
    # print(iou, sm.metrics.iou_score(y_true, y_pred))
    # print(dice, sm.metrics.f1_score(y_true, y_pred))

    fig = compare_segmentations(
            source_img,
            source_mask,
            predict,
            iou,
            dice,
            )

    # return the mask as an image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

    return f"""
    <!doctype html>
    <title>Predict display</title>
    <h1>Display result</h1>
    {pic_id}
    """


# ########## START BOTH API & FRONTEND ##########


if __name__ == "__main__":
    current_port = int(os.environ.get("PORT") or 5001)
    app.run(debug=True, host="0.0.0.0", port=current_port)
