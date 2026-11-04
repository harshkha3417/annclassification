# 🏦 Customer Churn Prediction using Artificial Neural Networks (ANN)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end Machine Learning and Deep Learning system designed to predict bank customer churn. This project utilizes an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras**, complete with exploratory data analysis, hyperparameter tuning, model persistence, and an interactive **Streamlit web application** for real-time inference.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset Overview](#-dataset-overview)
- [Model Architecture & Training](#-model-architecture--training)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [1. Running the Streamlit Web Application](#1-running-the-streamlit-web-application)
  - [2. Running the Notebooks](#2-running-the-notebooks)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Bonus: Salary Regression](#-bonus-salary-regression)
- [License](#-license)

---

## 📖 Overview

Customer churn occurs when customers stop doing business with a company. In the banking sector, retaining existing customers is significantly more cost-effective than acquiring new ones. 

This repository provides:
1. **ANN Classification Model**: Predicts the likelihood (`0` to `1`) of a customer leaving the bank based on demographics, financial history, and account activity.
2. **Interactive Streamlit Web Dashboard**: Allows non-technical stakeholders (e.g., bank managers, analysts) to input customer parameters and receive immediate churn risk assessment.
3. **Hyperparameter Optimization**: Systematic search for optimal hidden layer configurations, activation functions, and neuron counts using `GridSearchCV` and `KerasClassifier`.
4. **Salary Regression**: An ANN regression experiment predicting customer estimated salary from financial attributes.

---

## ✨ Key Features

- **End-to-End Pipeline**: From raw tabular data to feature encoding, scaling, training, evaluation, and live deployment.
- **Deep Neural Network**: Multi-layer perceptron architecture with ReLU activation, Dropout, and Sigmoid output.
- **Production Preprocessing Artifacts**: Serialized encoders (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`) and scaler (`scaler.pkl`) ensure zero data leakage and consistent transformation during inference.
- **Interactive UI**: Real-time customer churn probability slider and classification warning thresholds in Streamlit.
- **Training Telemetry**: Integrated TensorBoard callbacks and Early Stopping to avoid overfitting.

---

## 🏗 Project Architecture

```
┌────────────────────────────────┐
│      Churn_Modelling.csv       │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   Data Preprocessing Pipeline  │
│  - Label Encoding (Gender)     │
│  - One-Hot Encoding (Geography)│
│  - StandardScaler Normalization│
└───────────────┬────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│  Serialized  │ │  ANN Model   │
│ Artifacts    │ │  (model.h5)  │
│ (.pkl files) │ │              │
└───────┬──────┘ └──────┬───────┘
        │               │
        └───────┬───────┘
                │
                ▼
┌────────────────────────────────┐
│     Streamlit Web Service      │
│          (app.py)              │
│  - Input Customer Metrics      │
│  - Real-Time Churn Probability │
└────────────────────────────────┘
```

---

## 📂 Repository Structure

```
├── Churn_Modelling.csv           # Raw dataset containing 10,000 customer records
├── app.py                        # Streamlit web application for real-time churn prediction
├── model.h5                      # Pretrained Keras Artificial Neural Network model
├── label_encoder_gender.pkl      # Pickled LabelEncoder for gender transformation
├── onehot_encoder_geo.pkl        # Pickled OneHotEncoder for geography transformation
├── scaler.pkl                    # Pickled StandardScaler for feature normalization
├── experiments.ipynb             # Model development, EDA, training, and evaluation
├── hyperparametertuningann.ipynb # Hyperparameter tuning using GridSearchCV & KerasClassifier
├── prediction.ipynb              # Standalone inference & testing script on new data
├── salaryregression.ipynb        # ANN regression experiment predicting EstimatedSalary
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore file (excludes venv, checkpoints, caches)
└── README.md                     # Comprehensive project documentation
```

---

## 📊 Dataset Overview

The dataset (`Churn_Modelling.csv`) contains information on **10,000 bank customers** with the following attributes:

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Integer | Customer credit score |
| `Geography` | Categorical | Country of residence (France, Germany, Spain) |
| `Gender` | Categorical | Male or Female |
| `Age` | Integer | Age of customer |
| `Tenure` | Integer | Number of years the customer has been with the bank |
| `Balance` | Float | Account balance |
| `NumOfProducts`| Integer | Number of bank products used (accounts, loans, cards) |
| `HasCrCard` | Binary (0/1) | Whether the customer holds a credit card |
| `IsActiveMember`| Binary (0/1) | Customer activity status |
| `EstimatedSalary`| Float | Estimated annual salary |
| **`Exited`** | Binary (0/1) | **Target Variable**: 1 if customer churned, 0 otherwise |

*(Customer identifiers such as `RowNumber`, `CustomerId`, and `Surname` are dropped during preprocessing.)*

---

## 🧠 Model Architecture & Training

The classification model is a Sequential Artificial Neural Network built using TensorFlow/Keras:

- **Input Dimension**: Scaled numeric features + encoded categorical variables (11 input dimensions).
- **Hidden Layers**:
  - Dense Layer 1: 64 neurons, ReLU activation
  - Dense Layer 2: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation function (outputting churn probability between 0 and 1).
- **Loss Function**: Binary Cross-Entropy (`binary_crossentropy`).
- **Optimizer**: Adam.
- **Regularization & Callbacks**:
  - `EarlyStopping` monitoring validation loss with patience to prevent overfitting.
  - `TensorBoard` callback for tracking learning curves.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone git@github.com:harshkha3417/annclassification.git
cd annclassification
```

### 3. Set Up Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Running the Streamlit Web Application

Launch the interactive web UI locally:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. You can:
1. Select Geography (France, Germany, Spain) and Gender.
2. Adjust Age, Balance, Credit Score, Salary, Tenure, and Number of Products.
3. Toggle Active Member and Credit Card status.
4. View real-time churn probability and classification decision.

### 2. Running the Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

- **`experiments.ipynb`**: Complete exploratory data analysis, data encoding, training the ANN, and generating the serialized `.pkl` and `.h5` files.
- **`hyperparametertuningann.ipynb`**: Explore grid search experiments with SciKeras / KerasClassifier across layer depths and neuron sizes.
- **`prediction.ipynb`**: Demonstrates inference on unseen customer sample records.
- **`salaryregression.ipynb`**: Explores deep learning regression to predict continuous salary values.

---

## 🔬 Hyperparameter Tuning

Hyperparameter optimization is performed using `GridSearchCV` paired with `KerasClassifier`:
- **Evaluated parameters**: Number of hidden layers, neuron counts (16, 32, 64), learning rates, batch sizes, and epochs.
- Cross-validation ensures the network avoids overfitting on specific test folds.

---

## 📈 Bonus: Salary Regression

In addition to classification, this project explores an ANN regression problem in `salaryregression.ipynb`:
- **Target**: `EstimatedSalary` (continuous numerical target).
- **Architecture**: Dense layers with Linear output activation.
- **Evaluation Metric**: Mean Squared Error (MSE) and Mean Absolute Error (MAE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
