# EEG Language Decoding from Spectrogram Data (IRP Dissertation)

---

## Dissertation Report

### Download Full PDF  
[dissertation.pdf](./Gregory_Seaton_IRP.pdf)

---

## Overview

This project investigates whether EEG signals contain sufficient information to decode linguistic structure and meaning.

Multiple machine learning approaches were explored to map EEG spectrogram data to linguistic representations, including:
- Part-of-speech style classification  
- Word embedding prediction  
- Unsupervised structure discovery  
- Downstream classification using predicted embeddings  

The goal was to evaluate how far EEG signals can support semantic and syntactic language decoding.

---

## Key Methodologies

---

### 1. Content vs Function Word Classification

A neural network was trained to classify EEG signals based on whether a participant was reading a content or function word.

- Training time reduced from 28:26 to 1:46
- Improved stability of accuracy across runs
- Best accuracy: 0.66
- Did not exceed prior benchmark performance

---

### 2. Predicting Word Embeddings from EEG

A regression model was trained to predict continuous word embeddings from EEG signals.

- Cosine similarity: 0.67
- Statistically significant improvement over random baseline (t-test)
- Indicates EEG contains measurable semantic signal
- Embedding fidelity remains limited

---

### 3. Clustering Word Embedding Space

Unsupervised clustering methods were applied to pretrained GloVe embeddings.

- No meaningful cluster structure identified
- Suggests limited separability using standard clustering methods

---

### 4. Classification Using Predicted Embeddings

Predicted embeddings were evaluated on downstream classification tasks.

- Best accuracy: 0.57
- True embedding baseline: 0.96–0.99
- Confirms strong separability in embedding space
- Predicted embeddings retain partial linguistic structure

---

## Key Findings

### Noise and Model Sensitivity
- EEG data is highly noisy and sensitive to hyperparameters
- Over-parameterisation leads to overfitting

---

### Efficiency vs Performance Trade-off
- Spectrogram to CSV conversion significantly improved computational efficiency
- Introduced trade-offs in interpretability and performance

---

### Linguistic Context Matters
- Word ambiguity limits embedding prediction accuracy
- Contextual embeddings would likely improve performance
- Part-of-speech classification is less affected due to structural consistency

---

### Data Limitations
- EEG has a low signal-to-noise ratio
- More advanced modalities (e.g. fMRI) may improve results
- Requires new datasets for further exploration

---

## Reflections

While not all approaches achieved strong predictive performance, results consistently indicate that EEG signals contain statistically meaningful linguistic information.

Evidence for this includes:
- Significant improvement over random baselines in embedding prediction
- Strong separability in true embedding space classification tasks

---

## Exploratory Work

Several extensions (including clustering and embedding-based classification pipelines) were explored to broaden the scope of the project.

While not all were successful, they provided insight into:
- Limitations of EEG-based decoding
- Structure of linguistic embedding spaces
- Importance of focusing model complexity in noisy datasets

---

## Technologies

- Python
- PyTorch / TensorFlow
- Scikit-learn
- EEG spectrogram processing pipeline
- GloVe word embeddings
