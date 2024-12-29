# Spam Email Detector Using Random Forest

## Introduction
In this project, we developed a spam email detection system using machine learning techniques. The goal was to accurately classify emails as spam or non-spam (ham) using the Random Forest algorithm. The dataset used for this project was obtained from the UCI Machine Learning Repository and is known as the "Spambase" dataset.

## Dataset
The dataset consists of 5572 emails, each represented by 1000 features extracted using TF-IDF (Term Frequency-Inverse Document Frequency). These features capture the importance of specific words and characters in the emails. The dataset is balanced, with approximately 13.4% of the emails labeled as spam.

## Methodology

### Data Preprocessing
1. **Data Importation:** The data was imported from a JSON file named `spamjson.json` using Pandas.
2. **Text Cleaning:** The text data was cleaned by converting all text to lowercase, removing punctuation, and eliminating stop words.
3. **TF-IDF Vectorization:** The text data was converted into numerical features using TF-IDF vectorization with 1000 features.

### Model Building
1. **Model Selection:** We chose the Random Forest algorithm for its robustness and ability to handle large datasets with high dimensionality.
2. **Hyperparameter Tuning:** The Random Forest model was tuned with the following hyperparameters:
    - `n_estimators=200`
    - `max_depth=20`
    - `min_samples_split=5`
    - `min_samples_leaf=2`
    - `class_weight='balanced'`
    - `random_state=42`

### Model Training and Evaluation
1. **Train-Test Split:** The dataset was split into 80% for training and 20% for testing.
2. **Model Training:** The Random Forest model was trained on the training data.
3. **Model Evaluation:** The model was evaluated on the test data, yielding a high accuracy and favorable precision, recall, and F1-score metrics.

## Results
- **Accuracy:** The model achieved a high accuracy on the test set, indicating its effectiveness in spam detection.
- **Feature Importance:** The model identified the most important features (words) that contribute to spam classification, providing insights into which words are most indicative of spam.

## Conclusion
This project successfully demonstrated the application of machine learning in spam email detection. The Random Forest model performed well, accurately classifying emails as spam or ham. Future work could involve exploring other algorithms or enhancing feature engineering to further improve performance.

## References
- Dada, Emmanuel Gbenga, et al. "Machine learning for email spam filtering: review, approaches and open research problems." Heliyon 5.6 (2019).
- Karim, Asif, et al. "A comprehensive survey for intelligent spam email detection." Ieee Access 7 (2019): 168261-168295.
- Kaddoura, Sanaa, et al. "A systematic literature review on spam content detection and classification." PeerJ Computer Science 8 (2022): e830.
- Jain, Ankit Kumar, Sumit Kumar Yadav, and Neelam Choudhary. "A novel approach to detect spam and smishing SMS using machine learning techniques." International Journal of E-Services and Mobile Applications (IJESMA) 12.1 (2020): 21-38.
