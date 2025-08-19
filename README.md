# 🛒 Sentiment Analysis on Amazon Product Reviews  

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)  
[![Internship](https://img.shields.io/badge/Internship-SkilledScore.com-green)](https://skilledscore.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

## 📌 Project Overview
This repository contains my **Data Science Internship Project** completed at **[SkilledScore.com](https://skilledscore.com/)**.  

The goal of the project is to perform **Sentiment Analysis on Amazon Product Reviews** and compare the performance of **Machine Learning (Naive Bayes)** vs **Deep Learning (LSTM)** approaches.

---

## ⚙️ Features
- **Data Simulation** with `Faker` library (mock Amazon reviews)
- **Exploratory Data Analysis (EDA)** – distributions & word clouds
- **Preprocessing** – text cleaning, stopword removal, tokenization
- **Models**  
  - 📊 Naive Bayes with TF-IDF  
  - 🔁 LSTM with embeddings  
- **Evaluation** – Classification Reports & Confusion Matrices
- **Final Comparative Report**

---

## 🛠️ Tech Stack
- **Language:** Python 🐍  
- **Libraries:** Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, NLTK, WordCloud  

---

## 📑 Final Report & Conclusion  

### 🔍 Model Performance Summary  
We compared two models on the **Amazon Product Reviews Sentiment Analysis** task:  

1. **Naive Bayes (Tuned with TF-IDF)**  
   - Accuracy: **32%**  
   - **Strength:** Slightly better than chance level, recall = 0.56 for neutral class.  
   - **Weakness:** Struggled with positive & negative classes (low recall).  

2. **LSTM (Deep Learning Model)**  
   - Accuracy: **33%**  
   - **Strength:** Very high recall (1.0) for positive reviews.  
   - **Weakness:** Completely failed to capture neutral and negative classes.  

---

### 📊 Key Insights  
- Both models achieved accuracy around **32–33%**, close to random guessing.  
- **Naive Bayes** gave more balanced results, but still weak.  
- **LSTM** overfitted to the positive class due to class imbalance.  

---

### ⚠️ Reasons for Low Performance  
1. **Synthetic Data** – Reviews and sentiments were randomly generated → no real relationship.  
2. **Class Imbalance** – Random generation introduced bias.  
3. **Shallow Training** – Few epochs, no embeddings (like GloVe/Word2Vec) or tuning.  

---

### ✅ Conclusion  
- The purpose was to implement an **end-to-end sentiment analysis pipeline** (preprocessing → training → evaluation).  
- Both **Naive Bayes** and **LSTM** were successfully built and compared.  
- With real-world Amazon data, performance would improve with:  
  - High-quality labeled datasets  
  - Hyperparameter tuning  
  - Pre-trained embeddings or transformer models (e.g., **BERT**)  

📢 **Final Note:**  
This project demonstrates the **workflow of building and comparing sentiment analysis models**.  
The low performance is due to **synthetic data** and should not be considered a limitation of the models themselves.  

---

## 🙌 Acknowledgment
This project was completed as part of the **Data Science Internship Program** at **[SkilledScore.com](https://skilledscore.com/)**.  
Special thanks to the SkilledScore team for guidance and support.  

---

## 🚀 How to Run
1. Clone the repo:  
   ```bash
   git clone https://github.com/AsadullahShehbaz/SkilledScore-DS-Internship-Sentiment-Analysis.git
   cd SkilledScore-DS-Internship-Sentiment-Analysis
   ```
