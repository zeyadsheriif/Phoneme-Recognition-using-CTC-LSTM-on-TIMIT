# Phoneme-Recognition-using-CTC-LSTM-on-TIMIT
===========================================

Overview
--------

This repository contains a PyTorch implementation of an Acoustic-Phonetic Continuous Speech Recognizer. The model utilizes a Bidirectional Long Short-Term Memory (BiLSTM) network trained with Connectionist Temporal Classification (CTC) loss to directly map audio features to phoneme sequences.

The project is built and evaluated using the **DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus** and features custom, from-scratch implementations of CTC decoding algorithms.

Key Features
------------

*   **Automated Data Pipeline:** Integrates kagglehub to automatically download and extract the TIMIT dataset, with a built-in fallback to synthetic data generation for testing/debugging.
    
*   **Audio Preprocessing:** Extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs) using torchaudio (Sample rate: 16kHz, n\_fft: 400, hop\_length: 160, n\_mels: 23).
    
*   **Phoneme Dictionary:** Maps standard TIMIT phonemes (46 phonemes + token) to integer indices for CTC training.
    
*   **Custom Algorithms (From Scratch):**
    
    *   **CTC Forward Probability:** Computes the likelihood of a ground truth phoneme sequence given the model's logits.
        
    *   **Greedy Decoding:** Fast, argmax-based sequence prediction.
        
    *   **Beam Search Decoding:** Explores multiple sequence hypotheses (default beam width = 5) to improve recognition accuracy.
        
    *   **Levenshtein Distance:** Custom implementation to calculate Phoneme Error Rate (PER).
        

Model Architecture
------------------

The core model is a CTCLSTM built with PyTorch:

*   **Input Dimension:** 13 (N\_MFCC)
    
*   **Hidden Dimension:** 128
    
*   **RNN Layers:** 2
    
*   **Type:** Bidirectional LSTM (bidirectional=True)
    
*   **Output Layer:** Linear projection to the vocabulary size (47 classes).
    

Requirements
------------

Ensure you have Python 3.7+ installed. The primary dependencies are:

*   torch
    
*   torchaudio
    
*   numpy
    
*   matplotlib
    
*   soundfile
    
*   kagglehub
    

You can install the dependencies via pip:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install torch torchaudio numpy matplotlib soundfile kagglehub   `

Usage & Execution
-----------------

1.  Bashgit clone https://github.com/yourusername/timit-ctc-lstm.gitcd timit-ctc-lstm
    
2.  **Run the Notebook:**Open Zeyad.ipynb in Jupyter Notebook, JupyterLab, or Google Colab.
    
3.  **Training:**The notebook will automatically attempt to download the TIMIT dataset via the Kaggle API. It trains for **30 epochs** using the Adam optimizer (LR: 1e-3) with gradient clipping (max norm: 5.0) to stabilize CTC loss.
    
4.  **Evaluation:**During evaluation, the model outputs the Likelihood of the Ground Truth, the Target sequence, the Greedy prediction, and the Beam Search prediction.
    

Results
-------

*   **Training Convergence:** The CTC loss steadily decreases over 30 epochs (from ~0.69 down to ~0.29), demonstrating stable learning.
    
*   **Evaluation:** The model achieves an average **Phoneme Error Rate (PER) of ~0.32** on a subset of the test data using Beam Search decoding.
    
*   **Visualizations:** The notebook includes a heatmap visualization of the CTC Alignment Probabilities (Softmax Output) over time frames, highlighting how the model learns to output distinct phoneme spikes separated by blank tokens.
    

Author
------

**Zeyad Sherif** Developed as part of studies in advanced speech recognition, machine learning algorithms, and coursework for CSAI 498.
