# Voice-Based Stress Detection using Deep Learning

## Project Overview

This project focuses on detecting **stress levels from speech signals** using deep learning models. Speech contains temporal and acoustic patterns that reflect emotional states. The objective of the project is to classify speech recordings into **Low Stress** and **High Stress** categories.

The project is implemented as part of a **Deep Learning coursework scaffolded project** and consists of multiple stages evaluating different neural architectures.

---

## Dataset

We use the **RAVDESS Emotional Speech Audio Dataset**.

Dataset source:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

The dataset contains speech recordings from actors expressing different emotions.

For this project, emotions are mapped to stress levels:

Low Stress:

* Neutral
* Calm
* Sad

High Stress:

* Angry
* Fearful
* Disgust
* Surprised

---

## Project Structure

```
voice-stress-detection/
│
├── README.md
├── requirements.txt
│
├── Review1/
│   └── 25030-DL-Review1v5.ipynb
│
└── Review2/
    └── 25030-DL-Review2v6.ipynb
```

---

## Review 1: Feature Representation and Baseline Models

Implemented models:

* Multi-Layer Perceptron (MLP)
* Convolutional Neural Network (CNN)

Key components:

* Audio preprocessing
* MFCC feature extraction
* Mel spectrogram generation
* Data visualization (waveforms, MFCC, spectrograms)
* Class imbalance analysis
* Baseline model training
* Overfitting control (dropout, early stopping)

Evaluation metrics:

* Accuracy
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC

---

## Review 2: Temporal Modeling and Transfer Learning

Implemented models:

Sequence Models:

* RNN
* LSTM
* GRU
* Attention-based LSTM

Pretrained CNN Models:

* ResNet18
* MobileNetV2

Key techniques:

* Temporal sequence modeling
* Transfer learning
* Hyperparameter tuning
* Gradient clipping
* Early stopping
* Class weighted loss

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* ROC-AUC
* Confusion Matrix
* Classification Report
* ROC Curves

---

## Results Summary

Typical performance trend observed:

| Model       | Performance                                  |
| ----------- | -------------------------------------------- |
| RNN         | Baseline temporal model                      |
| LSTM        | Improved long-term dependency modeling       |
| GRU         | Efficient gated recurrent network            |
| ResNet18    | Strong performance on spectrogram features   |
| MobileNetV2 | Lightweight CNN with competitive performance |

---

## Reproducibility

Experiments are made reproducible using fixed random seeds for:

* Python random
* NumPy
* PyTorch

---

## Installation

Clone the repository:

```
git clone https://github.com/VK11-7/Voice-Based-Stress-Detection.git
cd voice-stress-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Notebooks

Open the notebooks using Jupyter or VS Code:

Review 1 notebook:

```
Review1/25030-DL-Review1v5.ipynb
```

Review 2 notebook:

```
Review2/25030-DL-Review2v6.ipynb
```

Run all cells sequentially.

---

## Author
Varadharajan K <br>
Deep Learning Coursework Project: Voice-Based Stress Detection
