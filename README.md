# 🎙️ Phoneme Recognition using BiLSTM-CTC on the TIMIT Dataset

An end-to-end Automatic Speech Recognition (ASR) system that converts speech into phoneme sequences using Bidirectional Long Short-Term Memory (BiLSTM) networks trained with Connectionist Temporal Classification (CTC) loss. The project is implemented in PyTorch and evaluated on the DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus.

---

## 🚀 Overview

This project demonstrates a complete phoneme recognition pipeline, from raw speech preprocessing to sequence decoding.

Unlike traditional frame-level classifiers, the model directly learns the alignment between acoustic features and phoneme sequences using CTC, eliminating the need for frame-level annotations.

Several decoding algorithms were implemented entirely from scratch, providing a deeper understanding of modern speech recognition systems.

---

## ✨ Features

* Automatic Speech Recognition (ASR)
* Bidirectional LSTM Acoustic Model
* Connectionist Temporal Classification (CTC)
* MFCC Feature Extraction
* Custom Greedy Decoder
* Custom Beam Search Decoder
* Forward Probability Algorithm
* Phoneme Error Rate (PER) Evaluation
* Automatic TIMIT Dataset Download
* Synthetic Dataset Fallback

---

## 🏗️ System Pipeline

```text
Speech Audio
      │
      ▼
MFCC Feature Extraction
      │
      ▼
Bidirectional LSTM
      │
      ▼
Linear Projection
      │
      ▼
CTC Output Probabilities
      │
      ▼
Greedy / Beam Search Decoder
      │
      ▼
Predicted Phoneme Sequence
```

---

## 🧠 Model Architecture

* Input Features: 13 MFCC coefficients
* Bidirectional LSTM
* 2 Recurrent Layers
* Hidden Size: 128
* Fully Connected Output Layer
* CTC Loss
* Adam Optimizer

---

## 🛠 Technologies

* Python
* PyTorch
* Torchaudio
* NumPy
* Matplotlib
* SoundFile
* KaggleHub

---

## 📂 Project Structure

```text
Phoneme-Recognition-BiLSTM-CTC/
│
├── CTC.ipynb
├── requirements.txt
├── README.md
└── assets/
```

---

## 🔬 Custom Implementations

Instead of relying solely on library implementations, several core speech recognition algorithms were developed from scratch:

* CTC Forward Probability
* Greedy Decoding
* Beam Search Decoding
* Levenshtein Distance
* Phoneme Error Rate (PER)

---

## 📊 Results

* Stable CTC convergence over 30 training epochs
* Average Phoneme Error Rate (PER) ≈ **0.32**
* Successful phoneme sequence prediction using both Greedy and Beam Search decoding
* Visualization of CTC alignment probabilities across speech frames

---

## 🎯 Learning Outcomes

* Automatic Speech Recognition
* Deep Learning
* Sequence Modeling
* Bidirectional LSTMs
* Connectionist Temporal Classification
* Beam Search
* Acoustic Modeling
* Speech Signal Processing

---

## 🚀 Future Improvements

* Replace BiLSTM with Transformer or Conformer architectures
* Integrate Language Models for decoding
* Support word-level speech recognition
* Deploy as a real-time speech recognition API
* Train on larger multilingual speech datasets

---

## 👨‍💻 Author

**Zeyad Sherif**
