#SUPRESS WARNINGS
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


#MAIN CODE
import cv2
import numpy as np
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
IMG_W = 128
IMG_H = 32

CHAR_LIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'\"-()"

model = tf.keras.models.load_model("trained model/model.keras", compile=False)

char_to_num = layers.StringLookup(
    vocabulary=list(CHAR_LIST), mask_token=None
)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True,
    mask_token=None
)

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0]

    output_text = []
    for res in results:
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(text)

    return output_text

#LOADING TEST IMAGE
img = cv2.imread("testing images/test 3.png", cv2.IMREAD_GRAYSCALE)
h, w= img.shape

if img is None:
    raise ValueError("Image not found or cannot be read")

img_ = cv2.resize(img, (IMG_W, IMG_H))
img_ = img_ / 255.0
img_ = img_[np.newaxis, ..., np.newaxis]

#PREDICTION
pred = model.predict(img_)

text = decode_prediction(pred)[0]
text = text.replace("[UNK]", "")

print(text)

#Adding text in image
space_h = 50  #Extra spacing for text area

new_img = np.zeros((h + space_h, w), dtype=np.uint8)
new_img[0:h, :] = img
new_img[h:h+space_h, :] = 255
cv2.putText(
    new_img,
    text,
    (10, h + 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    0,
    2,
    cv2.LINE_AA
)

cv2.imwrite('output_images/out 3.png', new_img)
cv2.imshow('Frame', new_img)
cv2.waitKey(0)