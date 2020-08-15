from __future__ import print_function, division
import os
import io
import json
import dlib
import torch
import base64
from io import BufferedReader, BytesIO
from skimage import io, color
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from xception import Xception


app = Flask(__name__)
device = torch.device('cpu')
model = Xception()
ckpt_dir = 'log_path/Xception_trained_model.pth'
checkpoint = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
message = ''


def crop_image(file):
    detector_ori = dlib.get_frontal_face_detector()
    # open the image file
    try:
        img = io.imread(file)
    except Exception as e:
        message = "While processing, " + str(e)
        return message

    # If the resolution is less than 128x128 then skip
    img_height = img.shape[0]
    img_width = img.shape[1]
    if img_height < 128 or img_width < 128:
        message = "While processing, image size too small"
        return message

    # find one face that best matches and finalize the image cropping size
    max_object = None

    dets, score, idx = detector_ori.run(img, 1, -1)
    max_confi = 0.6

    if len(dets) == 0:
        message = "While processing, face not detected"
        return message

    for i, d in enumerate(dets):
        if max_confi < score[i]:
            max_confi = score[i]
            max_object = d

    d = max_object

    if d == None:
        message = "While processing, face not detected"
        return message
    d_width = int((d.right() - d.left() + 1) // 2)
    d_height = int((d.bottom() - d.top() + 1) // 2)

    crop_top = d.top() - d_height
    crop_bottom = d.bottom() + d_height
    crop_left = d.left() - d_width
    crop_right = d.right() + d_width

    img_out_lenght = min(crop_top, crop_left, img_height -
                         crop_bottom, img_width - crop_right)

    if img_out_lenght < -d_width / 2:
        message = "While processing, face image over index"
        return message

    if img_out_lenght < 0:
        crop_top = crop_top - img_out_lenght
        crop_bottom = crop_bottom + img_out_lenght
        crop_left = crop_left - img_out_lenght
        crop_right = crop_right + img_out_lenght

    # Make the cropped and resized image from the original one
    img = img[crop_top:crop_bottom, crop_left:crop_right]
    if img.shape[0] != img.shape[1]:
        final_size = min(img.shape[0], img.shape[1])
        img = img[:final_size, :final_size]
    img = dlib.resize_image(img, 128 / img.shape[0])

    return img


def transform(im):
    gray_image = color.rgb2gray(im)
    gray_image = np.expand_dims(gray_image, -1)
    gray_image = gray_image.transpose((2, 0, 1))
    gray_image = torch.from_numpy(gray_image).view(1, 1, 128, 128)
    return gray_image


def test_im(img):
    im = transform(img)
    with torch.no_grad():
        input_image = im.cpu().float()
        # compute output
        output = model(input_image).cpu().numpy()
        a = output.tolist()
    return a


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form["image"]
        
        image = Image.open(BytesIO(base64.b64decode(data)))
        image.load()
        filename = secure_filename("tmp.jpeg")
        
        converted_image = Image.new("RGB", image.size, (255, 255, 255))
        if len(image.split()) == 3:
            converted_image.paste(image)
        else:
            converted_image.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        converted_image.save(filename)

        cropped = crop_image(filename)

        os.remove(filename)

        if type(cropped) == str:
            return jsonify({'error': cropped})
        else:
            output = test_im(cropped)
            real = output[0][0]
            fake = output[0][1]
            return jsonify({'real': real, 'fake': fake})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
