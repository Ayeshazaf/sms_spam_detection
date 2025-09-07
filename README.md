# SMS Spam Detector

This project implements a robust machine learning pipeline to classify SMS messages as **spam** or **ham** using both classical machine learning models and transformer-based deep learning. It demonstrates the comparison between traditional algorithms and state-of-the-art NLP models on the same dataset.

## Models Used

- **Logistic Regression**
- **Naive Bayes**
- **DistilBERT (Transformer Model)**

## Dataset

- ~5,500 SMS messages labeled as spam or ham.
- Comprehensive preprocessing: cleaning URLs, mentions, hashtags, and special characters.

## Pipeline Overview

1. **Data Collection & Preprocessing**
   - Cleaned text (removed URLs, mentions, hashtags, special characters).
   - Stratified split into train/validation/test sets to prevent data leakage.

2. **Classical Machine Learning**
   - **Feature Extraction:** TF-IDF vectorization.
   - **Algorithms:** Naive Bayes & Logistic Regression.
   - **Hyperparameter Tuning:** Performed with GridSearchCV.

3. **Transformer Model**
   - Fine-tuned DistilBERT with Hugging Face Trainer for text classification.

4. **Evaluation Metrics**
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1 Score**

## Results

| Model               | Accuracy   |
|---------------------|:----------:|
| Logistic Regression | ~99.3%     |
| Naive Bayes         | ~96%       |
| DistilBERT          | ~99.5%     |

## Key Highlights

- **Classical ML vs. Transformers:** Shows the performance gap and strengths of each approach on the same dataset.
- **End-to-End Pipeline:** From data cleaning to model evaluation.
- **Modern NLP:** Demonstrates the power of transfer learning in text classification.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayeshazaf/sms_spam_detection.git
   cd sms_spam_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train & evaluate models**
   - Run scripts for classical ML (`run_classical_ml.py`)
   - Run transformer pipeline (`run_transformer.py`)


## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

## License

This project is licensed under the MIT License.
