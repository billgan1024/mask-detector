from flask import Flask, request
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("model.h5")
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
# model.summary()

# rest api that takes in 28x28 images of files and outputs the number it represents from 0-9 
# works with multiple file uploads
@app.route("/", methods=["POST"])
def home():
    ret = {}
    for name, file in request.files.items():
        # update dictionary with filename + prediction
        # convert image to greyscale
        img = Image.open(file).convert("RGB").resize((180, 180)) 
        img_array = np.array(img)
        ret[file.filename] = class_names[int(tf.argmax(model.predict(img_array.reshape(1, 180, 180, 3)), axis=1))]
        # ret[file.filename] = tf.argmax(model.predict(img_array.reshape(1, 180, 180, 3)), axis=1)
    # flask converts a dict to a json object automatically 
    return ret 
