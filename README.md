# Handwritten Text Recognition

This project implements a **Handwritten Text Recognition (HTR)** system using **TensorFlow/Keras**.  
It trains a deep learning model using IAM dataset and predicts the corresponding text.

The architecture uses **CNNs for feature extraction**, **Bidirectional LSTMs for sequence learning**, and **CTC loss** for alignment-free transcription.

---

## Features

- Word-level handwritten text recognition
- CNN + BiLSTM + CTC-based architecture
- Trained on the **IAM Handwriting Dataset**
- Supports uppercase, lowercase, digits, and punctuation
- Displays predicted text directly on the image

---

## 🤖Model Architecture

- **Input**: Grayscale image `(32 × 128)`
- **Convolutional Layers**: Feature extraction
- **Bidirectional LSTM (×2)**: Sequence modeling
- **Dense + Softmax**: Character probabilities
- **CTC Loss**: Handles variable-length labels

---

## 📈Dataset
IAM Words dataset used.<br>https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database

---
## ©️Character Set
**abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789.,' "-()**

---

## 🚀Requirements
- Python 3.10
- Tensorflow
- OpenCV
- Numpy
- Keras
- IAM Words Dataset

**Suggested to use venv for project**
#### venv setup
```bash
py -3.10 -m venv venv
venv\Scripts\Activate.ps1
venv\Scripts\activate
python -m pip install --upgrade pip
```

#### Installing Libraries
```bash
python -m pip install tensorflow opencv-python numpy
```

## 📝Code Usage and Output

Run **train.py** and generate **model.keras**
This is the reference file for **Handwritten to text** model
Model Stored in - trained model/model.keras
```bash
python train.py
```

Run **main.py** to test the model
Image loaded from **testing images** folder
passed through prediction model

```bash
python main.py
```

#### Input vs Output

| Input | Output |
|------|--------|
| <img src="testing images/test 1.png" width="300"> | <img src="output_images/out 1.png" width="300"> |

**Input Image Path:** `testing images/test 1.png`  
**Output Image Path:** `output_images/out 1.png`

| Input | Output |
|------|--------|
| <img src="testing images/test 2.png" width="300"> | <img src="output_images/out 2.png" width="300"> |

**Input Image Path:** `testing images/test 2.png`  
**Output Image Path:** `output_images/out 2.png`

| Input | Output |
|------|--------|
| <img src="testing images/test 3.png" width="300"> | <img src="output_images/out 3.png" width="300"> |

**Input Image Path:** `testing images/test 3.png`  
**Output Image Path:** `output_images/out 3.png`
