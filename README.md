1. Introduction
Machine learning has become an essential tool in healthcare for building diagnostic and prognostic models. Traditionally, most models rely on a single data modality (e.g., only medical images or only tabular patient records). However, in real-world clinical practice, doctors consider multiple sources of information simultaneously—such as images, patient history, and demographics, before making decisions.
To bridge this gap, Multimodal Machine Learning (MML) combines diverse data types into a single framework. This approach not only improves prediction accuracy but also enhances trust, interpretability, and clinical applicability. In this project, we focus on automated ensemble multimodal learning that integrates structured clinical data (tabular features) with medical imaging. We use the PAD-UFES-20 dataset for skin lesion diagnosis as our case study.
2. Techniques Used to Implement the Model
The implementation leverages a combination of AutoML, deep learning, and fusion strategies:
●	AutoML (Automated Machine Learning):
1.	Automates the design of ML pipelines (preprocessing, model selection, hyperparameter tuning).
2.	Ensures robust performance without requiring expert manual tuning.
3.	Uses ensemble learning to combine multiple models for improved accuracy.

●	Unimodal Models:
1.	Tabular Data: Models like Logistic Regression, Random Forest, XGBoost, and deep learning models (MLP, FT-Transformer, TANGOS).
2.	Image Data: CNN architectures (ResNet, EfficientNet, MobileNet) and Vision Transformers (ViT, DINOv2).

●	Multimodal Fusion Strategies:

1.	Late Fusion – Combine predictions from unimodal models (decision-level).
2.	Early Fusion – Concatenate tabular + image features into a single representation.
3.	Joint Fusion – Learn modality-specific features and fuse them in an end-to-end fashion.

●	Explainable AI (XAI): SHAP values, Integrated Gradients, and example-based reasoning to understand model predictions.
●	Uncertainty Estimation: Conformal prediction framework to quantify confidence in predictions, guiding when more data (e.g., images) is needed.
3. About the Dataset (PAD-UFES-20)
https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
●	Source: Collected in Brazil from 1,373 patients.
●	Size: 2,298 skin lesion images.
●	Modalities:
○	Images: Smartphone photos of skin lesions.
○	Tabular Data: 21 clinical features including age, sex, lesion location, and symptoms (e.g., itching, bleeding, growth).

●	Labels (6-class classification):
○	Cancerous: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Melanoma.
○	Non-Cancerous: Actinic Keratosis (ACK), Melanocytic Nevus (NEV), Seborrheic Keratosis (SEK).

●	Binary classification task: Cancerous vs. Non-Cancerous.
This dataset is ideal for multimodal ML as it combines visual + structured information, mimicking real-world diagnosis.
4. Architecture of Implementation
The overall architecture follows the AutoPrognosis-M framework, which integrates tabular and imaging data through automated ML and ensemble methods:
1.	Input Stage
○	Image Data Pipeline: Images → Preprocessing (resize, normalization) → Feature extraction via CNN/ViT models.
○	Tabular Data Pipeline: Patient metadata → Imputation → Feature scaling/encoding → Classifier models.

2.	Fusion Strategies
○	Late Fusion: Predictions from image and tabular models combined (weighted average or stacking).
○	Early Fusion: Extracted image embeddings + tabular features concatenated → passed into a neural network classifier.
○	Joint Fusion: End-to-end model learns image + tabular features together for joint prediction.

3.	Ensemble Learning
○	Best-performing unimodal + multimodal models combined into a weighted ensemble.
○	Ensemble weights optimized using Bayesian optimization.

4.	Output Stage
○	Final prediction: lesion category (6-class) or cancerous/non-cancerous (binary).
Along with uncertainty estimation and interpretability results.
