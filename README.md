# Real-Time-Sign-Language-Recognition-system

A real-time AI-powered Sign Language Digit Recognition System developed using Machine Learning and Deep Learning techniques. This project recognizes hand gestures representing digits **0–9** through live webcam input and predicts the corresponding digit in real time.

Built as part of the **Artificial Intelligence and Machine Learning (IT2011)** module at the Sri Lanka Institute of Information Technology (SLIIT). 

---

## 📌 Project Overview

Communication barriers between hearing-impaired individuals and non-signers remain a major challenge in daily life. This project aims to reduce that gap by developing a real-time hand sign digit recognition system using AI models.

The system detects hand gestures from a webcam feed and classifies them into digits from **0 to 9**.

---

## 🎯 Objectives

* Develop a real-time sign language digit recognition system
* Train machine learning and deep learning models for gesture classification
* Achieve high classification accuracy under varying conditions
* Enable live webcam-based inference using OpenCV

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 📂 Dataset

Dataset used:

* **Sign Language Digits Dataset** from Kaggle
  [https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset)

### Dataset Details

* 10 classes (Digits 0–9)
* RGB hand sign images
* Approximately 2,000+ images
* Image size standardized for model training

---

## ⚙️ Data Preprocessing

The dataset underwent several preprocessing and cleaning steps:

* Image validation and duplicate removal
* Automatic hand annotation using a pre-trained detector
* Bounding-box cropping
* Image resizing to 128×128
* Blur detection and removal
* Darkness filtering
* Outlier removal using statistical techniques
* Stratified dataset splitting

These preprocessing steps improved data quality and model performance. 

---

## 🤖 Models Implemented

The following models were trained and evaluated:

* Convolutional Neural Network (CNN)
* Support Vector Machine (SVM)
* Multi-Layer Perceptron (MLP)
* Decision Tree
* K-Nearest Neighbors (KNN)

---

## 📊 Model Performance

| Model         | Accuracy |
| ------------- | -------- |
| SVM           | 95.93%   |
| MLP           | 93.31%   |
| CNN           | 93.22%   |
| Decision Tree | 68.09%   |
| KNN           | 54.23%   |

The CNN model was selected for real-time deployment due to its strong real-time image recognition capabilities and stable performance. 

---

## 🎥 Real-Time Detection

The trained CNN model was integrated with OpenCV to perform live sign detection through a webcam feed.

### Features

* Real-time hand detection
* Digit prediction (0–9)
* Bounding box visualization
* Prediction confidence display

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Real-Time-Sign-Language-Recognition-system.git
cd Real-Time-Sign-Language-Recognition-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python main.py
```

---

## 📁 Project Structure

```plaintext
Real-Time-Sign-Language-Recognition-system/
│
├── data/                # Dataset
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code           
├── README.md
└── requirements.txt
```

---

## ⚠️ Challenges Faced

* Real-time prediction speed optimization
* Handling lighting and background variations
* Dataset limitations under real-world conditions
* Domain gap between training images and webcam input

---

## 🔮 Future Improvements

* Improve robustness for real-world environments
* Add support for continuous gesture recognition
* Expand dataset diversity
* Improve performance under different lighting conditions
* Extend the system beyond digit recognition

---

## 👥 Team Members

* IT24104110 – Thulmanthi W.A.S
* IT24104081 – Gurusingha R.N
* IT24104133 – Vaaranan J
* IT24104068 – Chandrasiri R.M.D.S
* IT23379138 – Rajamuni R.D.V.R
* IT24610827 – Lahiruni K.L.M

---

## 📚 Academic Context

This project was developed for the **Artificial Intelligence and Machine Learning (IT2011)** module at SLIIT. 

---

## 📄 License

This project is developed for educational and research purposes.
