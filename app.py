from flask import Flask, request
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("model.h5")
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']

# this endpoint can work with multiple image uploads
@app.route("/", methods=["POST"])
def home():
    ret = {}
    for name, file in request.files.items():
        # resize image to 180x180x3
        img = Image.open(file).convert("RGB").resize((180, 180)) 
        img_array = np.array(img)
        # add it to the dictionary to be returned as a json object
        ret[file.filename] = class_names[int(tf.argmax(model.predict(img_array.reshape(1, 180, 180, 3)), axis=1))]
    return ret 
