*[Read this in English](README.md) | [Leggi in Italiano](README.it.md)*

# E.R.M.E.S. - Facial Expression Recognition System

**University of Salerno** **Course:** Machine Learning - A.Y. 2025/2026  
**Professor:** Giuseppe POLESE, Loredana CARUCCIO  

**Team:** * Ugo Manzo (Matricola: 0512119071) - [GitHub](https://github.com/UgoManzoED)
* Renato Natale (Matricola: 0512119641) - [GitHub](https://github.com/Re-1234)

> An R&D Computer Vision project for the automatic classification of human facial expressions, utilizing a custom Convolutional Neural Network (CNN) to decode emotional states.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-APACHE-blue)

---

## Table of Contents
* [About the Project](#about-the-project)
* [Features & Methodology](#features--methodology)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Benchmarking & Results](#benchmarking--results)
* [Future Developments & Model Limitations](#future-developments--model-limitations)
* [Disclaimer and Ethical Limits](#disclaimer-and-ethical-limits)
* [References](#references)

---

## About the Project
**E.R.M.E.S.** is an engineering endeavor aimed at extracting and mapping facial features to decode human emotions. The project follows an incremental approach that starts with evaluating the limits of classical Machine Learning (1D vectors), evolving into a custom 2D CNN (E.R.M.E.S. V1), validating interpretability via **Explainable AI (Grad-CAM)**, and testing domain information boundaries through **Transfer Learning (VGG16)**. 

The model was trained and benchmarked on the public **FER-2013** dataset.

## Features & Methodology
* **Custom CNN Architecture (E.R.M.E.S. V1):** A 5.87M parameter model specifically designed to process spatial geometry, outperforming classical machine learning baselines (62.00% Accuracy).
* **Experimental Setup:** The network was trained using the **Adam optimizer** to minimize the **Categorical Crossentropy** loss function for multi-class classification: $\mathcal{L}=-\sum_{i=1}^{C}y_i\log(\hat{y}_i)$
* **Explainable AI Integration:** Utilizes Grad-CAM to ensure decision transparency, proving the network extracts anatomically coherent patterns (e.g., anchoring on the glabella for *Angry* and *Disgust* classes).
* **Advanced Data Pipeline:** Implements an asynchronous `tf.data` pipeline with spatial Data Augmentation and dynamic class weights to mitigate severe class imbalance and structural noise.
* **Classical ML & Transfer Learning Benchmarks:** Includes comparative baselines using SVM/Random Forest (PCA) and VGG16 (with Upsampling) to mathematically demonstrate the theoretical bounds of the dataset.

## Built With
* [Python](https://www.python.org/)
* [TensorFlow / Keras](https://www.tensorflow.org/) (for CNN, `tf.data`, and TensorBoard)
* [Scikit-Learn](https://scikit-learn.org/) (for PCA, SVM, Random Forest)
* [Jupyter Notebook](https://jupyter.org/)

---

## Getting Started
Follow these instructions to get a local copy up and running on your machine.

### Prerequisites
You need Python installed on your system. It is highly recommended to use a virtual environment.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/UgoManzoED/E.R.M.E.S..git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the dataset:
   * Manually download the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).
   * Place it inside a `data/` folder in the project root.

---

## Usage
The project is structured into sequential Jupyter Notebooks. Run them in order to reproduce the exploratory data analysis, training, and evaluation phases.

```text
ERMES-Project/
 |-- notebooks/
 |   |-- 01_EDA_Data_Analysis.ipynb        # Statistical analysis and detection rates
 |   |-- 02_Training_CNN.ipynb             # tf.data ingestion, augmentation, class weights
 |   |-- 03_Model_Baseline.ipynb           # Flattening, PCA, SVM, and Random Forest
 |   |-- 04_Model_CNN.ipynb                # E.R.M.E.S. V1 architecture and training
 |   |-- 05_Explainable_AI_GradCAM.ipynb   # Predictive transparency via heatmaps
 |   |-- 06_Transfer_Learning_VGG16.ipynb  # Upsampling and VGG16 fine-tuning
```

*Note: Weight files (`*.h5`), TensorBoard log directories, and the original dataset are intentionally excluded from version control.*

## Benchmarking & Results

| Model / Architecture | Global Accuracy (%) | Macro F1-Score |
| :--- | :---: | :---: |
| Zero Rule Baseline (Majority Class) | - | 0.06 |
| Random Baseline (Stratified) | - | 0.14 |
| Linear SVM + PCA | 34.00 | 0.28 |
| Random Forest (Non-Linear Ensemble) | 46.00 | 0.44 |
| **E.R.M.E.S. V1 (Custom CNN)** | **62.00** | **0.60** |
| VGG16 (Upsampling + Fine-Tuning) | 62.00 | 0.61 |
| *Human Baseline (Theoretical Upper Bound)* | *~65.00±5* | *-* |

## Future Developments & Model Limitations
- [ ] **Demographic Bias:** Address demographic representation biases inherent in the FER-2013 dataset.
- [ ] **Robustness:** Improve model resilience to poor lighting conditions and physiological occlusions.
- [ ] **Explainability:** Expand the Explainable AI suite with additional feature attribution methods (e.g., SHAP, Integrated Gradients).

## Disclaimer and Ethical Limits
The E.R.M.E.S. model is designed exclusively as an academic IT research prototype and **does not possess any diagnostic or clinical validity**. System performance is intrinsically subject to demographic representation biases present in the training data and drops significantly under poor lighting conditions or in the presence of physiological occlusions.

## References
1. **FER-2013:** Goodfellow, I.J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests*.
2. **VGG16:** Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
3. **Grad-CAM:** Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.
