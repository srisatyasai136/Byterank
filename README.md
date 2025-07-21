# Byterank
# 🚀 ByteRank: Coding Skill Predictor

ByteRank is a Flask web app that predicts your coding skill level based on your performance across platforms like LeetCode, GeeksforGeeks, CodeChef, and HackerRank. It uses a trained machine learning model (Random Forest Classifier) to classify users into Beginner, Intermediate, or Advanced levels.

---

## 🔧 Features

- Predicts skill level using 4 inputs:
  - LeetCode problems solved
  - GeeksforGeeks problems solved
  - CodeChef rating
  - HackerRank badges
- Input validation (no negative values, no empty fields)
- Error messaging
- Neat UI with HTML/CSS
- Machine Learning model saved and reused via Pickle

---

## ⚙️ Tech Stack

- **Backend:** Flask, Python
- **ML:** scikit-learn, pandas, numpy
- **Frontend:** HTML/CSS (custom styled)
- **Deployment:** Render
- **Others:** Pickle for model storage, Gunicorn for WSGI

---

## 🖥️ Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/your-username/coding-skill-predictor.git
cd coding-skill-predictor

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
