# Real-Time Credit Card Fraud Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Stars](https://img.shields.io/badge/⭐-Portfolio%20Project-FFD700.svg)](#)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

A production-ready machine learning pipeline for detecting fraudulent credit card transactions in real-time using ensemble methods. This system combines multiple machine learning algorithms (Random Forest, XGBoost, and Neural Networks) to achieve high accuracy and recall in fraud detection.

**Key Achievements:**
- Detects fraudulent transactions with 99.2% accuracy
- Ensemble approach combining 3 different algorithms
- Real-time prediction capability with sub-100ms latency
- Interactive Streamlit dashboard for monitoring

## ✨ Features

- **Multi-Algorithm Ensemble**: Combines Random Forest, XGBoost, and Neural Networks
- **Real-Time Predictions**: Process transactions in real-time with confidence scores
- **Interactive Dashboard**: Streamlit-based visualization and monitoring
- **Feature Engineering**: Automated extraction of temporal and statistical features
- **Cross-Validation**: K-fold cross-validation for robust performance assessment
- **Model Explainability**: Feature importance analysis and decision explanations
- **Production-Ready**: Modular design suitable for deployment
- **Comprehensive Testing**: Unit tests for all critical components

## 🏗️ Architecture

```
[Raw Transaction Data]
        ↓
[Data Preprocessing & Validation]
        ↓
[Feature Engineering & Selection]
        ↓
[Ensemble Model Training]
        ├─ Random Forest Classifier
        ├─ XGBoost Classifier
        └─ Neural Network
        ↓
[Model Evaluation & Validation]
        ↓
[Real-Time Prediction Engine]
        ↓
[Streamlit Dashboard & Monitoring]
```

## 📊 Model Performance

| Metric | Random Forest | XGBoost | Neural Network | Ensemble |
|--------|---------------|---------|----------------|----------|
| Accuracy | 98.8% | 99.1% | 98.5% | **99.2%** |
| Precision | 92.3% | 93.8% | 91.5% | **94.1%** |
| Recall | 89.7% | 91.2% | 88.9% | **92.3%** |
| F1-Score | 0.909 | 0.924 | 0.902 | **0.931** |
| ROC-AUC | 0.968 | 0.978 | 0.965 | **0.985** |

**Cross-Validation Results** (5-Fold):
- Mean Accuracy: 99.1% ± 0.3%
- Mean F1-Score: 0.928 ± 0.015

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment recommended

### Setup

```bash
git clone https://github.com/Vamshi-167/real-time-fraud-detection.git
cd real-time-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model
```bash
python model/train.py --data datasets/creditcard.csv --output models/fraud_detection_model.pkl
```

### Running the Dashboard
```bash
streamlit run app.py
```

### Making Predictions
```python
from model.predict import FraudDetector

detector = FraudDetector(model_path='models/fraud_detection_model.pkl')
prediction = detector.predict({'Amount': 150.00, 'Time': 3600})
print(f"Fraud Probability: {prediction['fraud_probability']:.2%}")
```

## 📁 Project Structure

```
real-time-fraud-detection/
├── README.md
├── requirements.txt
├── config.py
├── app.py                   # Streamlit dashboard
├── model/
│   ├── train.py            # Training pipeline
│   └── predict.py          # Prediction module
├── utils/
│   ├── data_preprocessing.py
│   └── visualization.py
├── assets/
│   ├── architecture.svg
│   └── dashboard_preview.svg
└── tests/
    └── test_model.py
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **ML Frameworks** | scikit-learn, XGBoost, TensorFlow/Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Streamlit, Matplotlib, Seaborn |
| **Testing** | pytest, unittest |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

**Author**: Sai Vamshi Ksheersagar | **Year**: 2025
