# SMS Spam Detection

This repository contains code and resources for detecting spam messages in SMS texts using machine learning techniques. The project aims to classify incoming SMS messages as either "spam" or "ham" (not spam), leveraging various data preprocessing and classification algorithms.

## Features

- **Data Preprocessing:** Cleaning and transforming SMS text data for use in machine learning models.
- **Feature Extraction:** Using techniques like TF-IDF, Bag-of-Words, or custom text features.
- **Model Training:** Implementing and training models such as Naive Bayes, Logistic Regression, or other classifiers.
- **Evaluation:** Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.
- **Prediction:** Providing functionality to predict if an input SMS is spam or not.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ayeshazaf/sms_spam_detection.git
    cd sms_spam_detection
    ```

2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Dataset

The project typically uses publicly available SMS spam datasets. You can place your dataset in the `data/` directory or update the relevant data loading paths in the scripts.

## Project Structure

```
sms_spam_detection/
├── data/               # Dataset files
├── models/             # Saved model files
├── src/                # Source code for preprocessing, training, prediction
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.

## Acknowledgments

- UCI SMS Spam Collection dataset
- scikit-learn, pandas, numpy, and other open-source libraries
