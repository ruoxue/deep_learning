import re
import json
import base64
import tensorflow as tf
import PIL

import numpy as np
import tensorflow.python.keras as keras

model = tf.keras.models.load_model(
    "model.h5", custom_objects=None, compile=True, options=None
)
img_arr = tf.keras.preprocessing.image.load_img('output.png',
                                                target_size=(28, 28),
                                                color_mode="grayscale")

img = tf.keras.preprocessing.image.img_to_array(img_arr) / 255.

img = np.expand_dims(img, axis=0)
code = model.predict(img)
print(np.argmax(code, axis=1))
