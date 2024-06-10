from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
import cv2


app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def home():
    return jsonify({"message": "halo 2"})

@app.route("/upload", methods=['POST'])
@cross_origin()
def upload():
    file = request.files["image"]

    
    img_original = Image.open(file)
    img_original = img_original.resize((128, 128))
    img_original = img_original.convert('L')
    res_encode = encode_image(img_original)
    res_decode = decode_image(res_encode)

    buffered_original = BytesIO()
    img_original.save(buffered_original, format="JPEG")
    base64_original = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

    buffered_encode = BytesIO()
    rescaled_image = (np.maximum(res_encode,0)/res_encode.max()) * 255
    img_encode = np.uint8(rescaled_image)
    img_encode = np.squeeze(img_encode, axis=2)
    img_encode = Image.fromarray(img_encode)

    img_encode.save(buffered_encode, format="JPEG")
    base64_encode = base64.b64encode(buffered_encode.getvalue()).decode('utf-8')

    buffered_decode = BytesIO()
    rescaled_image = (np.maximum(res_decode,0)/res_decode.max()) * 255
    img_decode = np.uint8(rescaled_image)
    img_decode = np.squeeze(img_decode, axis=2)
    img_decode = Image.fromarray(img_decode)

    img_decode.save(buffered_decode, format="JPEG")
    base64_decode = base64.b64encode(buffered_decode.getvalue()).decode('utf-8')
    
    return jsonify({
        "message": "Gambar berhasil diunggah", 
        "images": [
        {
            "type": "original",
            "file": 'data:image/png;base64,' + base64_original,
            "name": file.filename,
            "size": '',
            "pixel": np.asarray(img_original).shape,
        },
        {
            "type": "encode",
            "file": 'data:image/png;base64,' + base64_encode,
            "size": '',
            "pixel": '',
        },
        {
            "type": "decode",
            "file": 'data:image/png;base64,' + base64_decode,
            "size": '',
            "pixel": '',
        }
    ]
})

    # return

if __name__ == "__main__":
    app.run()

dir_model = "1.6 Autoencoder"
model = load_model(dir_model)

def decode_image(img):
    rescaled_image = (np.maximum(img,0)/img.max()) * 255
    img_encode = np.uint8(rescaled_image)
    img_encode = np.squeeze(img_encode, axis=2)
    img_encode = Image.fromarray(img_encode)
    image_array = tf.keras.preprocessing.image.img_to_array(img_encode)
    image_array = tf.keras.preprocessing.image.smart_resize(image_array, (128, 128))
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = model.predict(image_array)
    return prediction[0]

def encode_image(img):
    encoder = Model(inputs=model.input, outputs=model.get_layer('conv2d_transpose').output)

    image_array = tf.keras.preprocessing.image.img_to_array(img)
    image_array = tf.keras.preprocessing.image.smart_resize(image_array, (128, 128))
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = encoder.predict(image_array)
    return prediction[0]
