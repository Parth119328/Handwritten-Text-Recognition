#SUPRESS WARNINGS
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import os
import cv2
import numpy as np
from tensorflow.keras import layers, models

# CONFIG (Can be changed to according to requirement)
IMG_W = 128
IMG_H = 32
BATCH_SIZE = 32
EPOCHS = 70

# Character set
CHAR_LIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'\"-()"

#DATA LOADING
def load_data(words_txt, words_dir):
    images = []
    labels = []

    with open(words_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            if len(parts) < 9:
                continue

            word_id = parts[0]
            status = parts[1]

            if status != "ok":
                continue

            text = " ".join(parts[8:])

            p1 = word_id.split("-")[0]
            p2 = "-".join(word_id.split("-")[:2])

            img_path = os.path.join(
                words_dir, p1, p2, word_id + ".png"
            )

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_W, IMG_H))
            img = img / 255.0

            images.append(img)
            labels.append(text)

    return np.array(images), labels

# LABEL ENCODING
char_to_num = layers.StringLookup(
    vocabulary=list(CHAR_LIST), mask_token=None
)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), invert=True, mask_token=None
)

def encode_labels(labels):
    encoded = []
    lengths = []

    for text in labels:
        chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
        label = char_to_num(chars)
        encoded.append(label)
        lengths.append(len(label))

    return encoded, np.array(lengths)

# LOADING DATA
images, texts = load_data("iam_words/words.txt", "iam_words/words")

images = images[..., np.newaxis]

labels, label_lengths = encode_labels(texts)

max_label_len = max(label_lengths)
labels_padded = tf.keras.preprocessing.sequence.pad_sequences(
    labels, maxlen=max_label_len, padding="post"
)

input_lengths = np.ones((len(images), 1)) * (IMG_W // 4)

# MAIN MODEL
image_input = layers.Input(shape=(IMG_H, IMG_W, 1), name="image")
label_input = layers.Input(shape=(max_label_len,), name="label")
input_len = layers.Input(shape=(1,), name="input_length")
label_len = layers.Input(shape=(1,), name="label_length")

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Reshape((IMG_W // 4, (IMG_H // 4) * 64))(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

output = layers.Dense(len(CHAR_LIST) + 1, activation="softmax")(x)

def ctc_loss(args):
    y_pred, labels, input_len, label_len = args
    return tf.keras.backend.ctc_batch_cost(
        labels, y_pred, input_len, label_len
    )

ctc_loss_layer = layers.Lambda(
    ctc_loss,
    output_shape=(1,),
    name="ctc"
)(
    [output, label_input, input_len, label_len]
)

model = models.Model(
    inputs=[image_input, label_input, input_len, label_len],
    outputs=ctc_loss_layer
)

model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: y_pred
)

#TO STOP MODEL TRAINING BEFORE IT REACHES STATE OF OVERFITTING
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

# MODEL TRAINING
model.fit(
    [images, labels_padded, input_lengths, label_lengths],
    np.zeros(len(images)),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop]
)


# SAVING REF MODEL
prediction_model = models.Model(image_input, output)
prediction_model.save("trained model/model.keras")

print("Model saved as 'model.keras'")