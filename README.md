# 📰 Fake News Detection with Machine Learning

This project detects whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) and classical ML models like Logistic Regression and Naive Bayes.

---

## 📂 Dataset

The dataset comes from [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), containing:

- `Fake.csv` — Fake news articles
- `True.csv` — Real news articles

---

## 🔍 Workflow

- Load & merge data
- Text preprocessing using NLTK
- TF-IDF vectorization
- Logistic Regression classification
- Evaluation via confusion matrix and metrics
- (Optional) Streamlit app for deployment

---

## 💻 How to Run

### Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
