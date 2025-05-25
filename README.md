# ⚠️ NLP Hate Speech Detection with DistilBERT

This project uses **DistilBERT**, a transformer-based language model, to detect **hate speech** in text data. It demonstrates the use of **transfer learning** via Hugging Face Transformers to achieve high accuracy on a real-world text classification problem.

---

## 📌 Project Overview

Hate speech detection is critical for maintaining safe and respectful online communities. In this project, we fine-tune a pre-trained **DistilBERT model** to classify text as either **hate speech** or **non-hate speech**, using labeled data.

The final model achieves high performance by leveraging a robust NLP pipeline and advanced neural architectures.

---

## 📁 Workflow

### 1. 🧹 Data Preprocessing
- Text normalization
- Emoji conversion and removal
- Lowercasing, punctuation cleanup
- Tokenization compatible with BERT

### 2. 🤖 Model: DistilBERT
- Used `distilbert-base-uncased` from Hugging Face
- Tokenization via `AutoTokenizer`
- Fine-tuning with `TFAutoModelForSequenceClassification`
- Batched data using `tf.data.Dataset` for GPU acceleration

### 3. ⚖️ Imbalance Handling
- Dataset was **highly imbalanced** (majority class = non-hate)
- Solutions implemented:
  - `class_weight` in training loop
  - Focal Loss (optional)
  - Stratified sampling during data splitting

### 4. 📊 Evaluation
- Metrics used: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Special focus on **minority class (hate speech)**
- Plotted Confusion Matrix and Precision-Recall curve

---

## ✅ Final Results

- **Model**: Fine-tuned DistilBERT
- **F1 Score (macro)**: ~0.82
- **ROC-AUC**: ~0.91
- Strong performance on both classes, with robustness on the minority (hate) class.

---

## 🧰 Technologies Used

- Python 3.x  
- Hugging Face Transformers (`transformers`)
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- `emoji` library for emoji normalization

---
