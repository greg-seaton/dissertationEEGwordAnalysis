# EEG Language Decoding from Spectrogram Data (IRP Dissertation)

## 📄 Full Dissertation Report
👉 [Download PDF Dissertation](./Gregory_Seaton_IRP (1).pdf)

---

## Overview

This Individual Research Project investigates whether EEG signals can be used to extract meaningful linguistic information, focusing on the relationship between brain activity and language processing.

The work explores multiple machine learning approaches using EEG spectrogram data to model linguistic features such as part-of-speech structure and word semantics.

---

## Key Methodologies

Four core experimental directions were explored:

### 1. Content vs Function Word Classification (POS-style task)
A neural network was trained to classify whether EEG data corresponds to content or function words.

- Achieved improved training efficiency (reduced training time from **28:26 → 1:46**)
- Improved stability of accuracy compared to previous implementations
- Maximum accuracy achieved: **0.66**
- Did not surpass prior benchmark performance

---

### 2. Predicting Word Embeddings from EEG
A model was trained to predict continuous word embeddings from EEG signals.

- Achieved cosine similarity of **0.67**
- Statistical testing (t-test) confirmed results were significantly better than random
- Indicates EEG contains meaningful semantic signal, though representation quality remains limited

---

### 3. Clustering Word Embedding Space
Unsupervised clustering methods were applied to pretrained GloVe embeddings.

- No meaningful cluster structure was identified
- Suggests limited separability in embedding space using standard clustering approaches

---

### 4. Classification Using Predicted Embeddings
Predicted embeddings were evaluated on downstream classification tasks.

- Best accuracy: **0.57**
- Strong baseline performance using true embeddings (**0.96–0.99 accuracy**) confirmed strong separability between content and function words
- Demonstrated that predicted embeddings retain partial linguistic signal

---

## Key Findings

### Noise and Model Complexity
- High-dimensional, noisy EEG data is highly sensitive to hyperparameter tuning
- Over-parameterisation led to overfitting in multiple experiments

### Representation Trade-offs
- Spectrogram-to-CSV conversion significantly improved computational efficiency
- However, this introduced trade-offs in interpretability and potentially reduced performance

### Contextual Limitations
- Word ambiguity limits embedding prediction accuracy
- Contextual embedding models could significantly improve performance
- This issue is less pronounced in part-of-speech classification

### Data Quality Constraints
- EEG’s high noise-to-signal ratio limits achievable accuracy
- Alternative modalities (e.g. fMRI) may improve feature quality but require new datasets

---

## Reflections

While several approaches did not achieve strong predictive performance, the experiments consistently demonstrated that EEG signals contain measurable linguistic information.

A key outcome of this work is the validation that:
- EEG carries statistically meaningful semantic signal
- Word embeddings provide a viable intermediate representation for decoding language from neural activity

---

## Notes on Scope

Several exploratory directions (e.g. clustering and embedding-based classification pipelines) were intentionally pursued to investigate alternative formulations of the problem. While not all yielded strong results, they contributed to a broader understanding of the limits of EEG-based language decoding.

---

## Technologies

- Python
- PyTorch / TensorFlow (if applicable)
- EEG spectrogram processing pipeline
- Scikit-learn
- NLP embeddings (GloVe)
