# 🎬 Movie Sentiment Analyzer

A **Deep Learning–based web application** that predicts the sentiment of movie reviews as **Positive** or **Negative**.
The system uses a **Recurrent Neural Network (RNN)** trained on the **IMDB movie review dataset** and is deployed using **Flask** for real-time predictions.

---

## 📌 Overview

Movie reviews often contain subjective opinions. This project uses **Natural Language Processing (NLP)** and **Deep Learning** to automatically classify a review’s sentiment.

The model processes the input text, converts it into numerical form, and predicts whether the sentiment is **positive** or **negative**.

---

## 🚀 Features

* Real-time sentiment prediction
* Deep learning model trained on IMDB dataset
* Simple and clean web interface
* Pretrained model included
* Lightweight Flask backend

---

## 🛠️ Tech Stack

| Category      | Technology                     |
| ------------- | ------------------------------ |
| Language      | Python                         |
| Backend       | Flask                          |
| Deep Learning | TensorFlow           |
| Model Type    | RNN (Recurrent Neural Network) |
| Frontend      | HTML, CSS                      |

---

## 🧠 Model Details

* Dataset: **IMDB Movie Reviews**
* Model: **Recurrent Neural Network (RNN)**
* Task: Binary sentiment classification
* Output:

  * Positive
  * Negative

---

## 📁 Project Structure

```
Movie-Sentiment-Analyzer/
│
├── app.py                  # Flask application
├── imdb_rnn_model.h5       # Trained RNN model
├── templates/
│   └── index.html          # Web interface
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ashwinmali7781/Movie-Sentiment-Analyzer.git
cd Movie-Sentiment-Analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install flask tensorflow
```

---

### 4. Run the application

```bash
python app.py
```

---

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## 👨‍💻 Author

**Ashwin Mali**
BTech CSE (AIML)

GitHub: https://github.com/ashwinmali7781

---

## 📜 License

This project is licensed under the **MIT License**.
