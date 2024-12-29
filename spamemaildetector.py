# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the dataset
print("Loading dataset...")
df = pd.read_json("spamjson.json")

# Data Preprocessing
df = df[['v1', 'v2']]  # Remove unnecessary columns
df.columns = ['label', 'text']  # Arrange column names
df['text'] = df['text'].astype(str)  # Convert texts to string
df['label'] = (df['label'] == 'spam').astype(int)  # Convert labels to 1/0

# First create length column
df['length'] = df['text'].apply(len)

# Data Analysis and Visualization
plt.figure(figsize=(15, 10))

# 1. Spam/Ham Distribution (Pie Chart)
plt.subplot(2, 2, 1)
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Normal', 'Spam'])
plt.title('Spam/Normal Email Distribution')

# 2. Text Length Distribution
plt.subplot(2, 2, 2)
df['text_length'] = df['text'].apply(len)
sns.histplot(data=df, x='text_length', hue='label', bins=50)
plt.title('Email Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

# 3. Most Frequently Used Words (Spam vs Normal)
def get_common_words(texts):
    words = []
    for text in texts:
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        words.extend(text.split())
    return Counter(words).most_common(10)

spam_words = get_common_words(df[df['label'] == 1]['text'])
normal_words = get_common_words(df[df['label'] == 0]['text'])

# Spam Words
plt.subplot(2, 2, 3)
words, counts = zip(*spam_words)
plt.barh(words, counts)
plt.title('Most Common Spam Words')
plt.xlabel('Frequency')

# Normal Words
plt.subplot(2, 2, 4)
words, counts = zip(*normal_words)
plt.barh(words, counts)
plt.title('Most Common Normal Words')
plt.xlabel('Frequency')

plt.tight_layout()
plt.show()

# Get stop words list
def get_top_words(texts, n=10):
    # Combine all texts and convert to lowercase
    words = ' '.join(texts).lower()
    
    # Remove punctuation
    words = re.sub(r'[^\w\s]', '', words)
    
    # Split into words
    words = words.split()
    
    # Remove stop words and numbers
    words = [word for word in words 
            if word not in ENGLISH_STOP_WORDS  # Remove stop words
            and not word.isdigit()  # Remove numbers
            and len(word) > 2]  # Remove words shorter than 2 characters
    
    return pd.Series(words).value_counts().head(n)

# Visualize most common words
plt.figure(figsize=(15, 6))

# Most common words in spam emails
plt.subplot(1, 2, 1)
spam_words = get_top_words(df[df['label'] == 1]['text'])
spam_words.plot(kind='bar', color='red', alpha=0.6)
plt.title('Most Common Words in Spam Emails\n(Excluding Stop Words)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# Most common words in normal emails
plt.subplot(1, 2, 2)
normal_words = get_top_words(df[df['label'] == 0]['text'])
normal_words.plot(kind='bar', color='blue', alpha=0.6)
plt.title('Most Common Words in Normal Emails\n(Excluding Stop Words)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Print most common words
print("\nTop 10 words in spam emails:")
print(spam_words)
print("\nTop 10 words in normal emails:")
print(normal_words)

# Model Training
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    lowercase=True
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Model Performance Visualization
y_pred = model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Top Features
feature_importance = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
})
top_features = feature_importance.nlargest(20, 'importance')

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y='word')
plt.title('Top 20 Features')
plt.xlabel('Importance')
plt.show()

# Statistical Summary and Model Performance Metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# 1. Statistical Summary
print("\nDataset Statistical Summary:")
stats_df = pd.DataFrame({
    'Metric': [
        'Total Emails',
        'Spam Emails',
        'Normal Emails',
        'Spam Ratio (%)',
        'Average Email Length',
        'Median Email Length',
        'Shortest Email',
        'Longest Email'
    ],
    'Value': [
        len(df),
        sum(df['label'] == 1),
        sum(df['label'] == 0),
        (sum(df['label'] == 1) / len(df)) * 100,
        df['length'].mean(),
        df['length'].median(),
        df['length'].min(),
        df['length'].max()
    ]
})
print(stats_df.to_string(index=False))

# 2. Model Performance Metrics Visualization
# Convert classification report to DataFrame
cr = classification_report(y_test, y_pred, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()

# Classification Report Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(cr_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.3f')
plt.title('Model Performance Metrics')
plt.show()

# 3. Precision-Recall Curve
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, 'b-', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend()
plt.show()

# 4. Feature Importance Word Analysis
# Top 20 important words and their importance
feature_importance = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Importance', y='Word')
plt.title('Top 20 Important Words in Spam Detection')
plt.xlabel('Importance')
plt.show()

# 5. Confusion Matrix in Detail
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add percentages on the confusion matrix
cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j+0.5, i+0.7, f'({cm_percentages[i,j]:.1%})', 
                ha='center', va='center')

plt.show()

# 6. Model Score Distribution
plt.figure(figsize=(10, 6))
scores = model.predict_proba(X_test)[:, 1]
sns.histplot(data=pd.DataFrame({
    'Score': scores,
    'Actual Label': y_test
}), x='Score', hue='Actual Label', bins=50)
plt.title('Model Prediction Score Distribution')
plt.xlabel('Spam Probability')
plt.ylabel('Frequency')
plt.show()
# Combine columns and TF-IDF features
print(f"Total columns in dataset: {3 + X.shape[1]}")

# Show first 10 TF-IDF features
print("\nFirst 10 TF-IDF features:")
print(vectorizer.get_feature_names_out()[:10])

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Detector")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Combobox for example emails
        ttk.Label(main_frame, text="Example emails:").grid(row=0, column=0, sticky=tk.W)
        self.example_var = tk.StringVar()
        self.example_combo = ttk.Combobox(main_frame, textvariable=self.example_var)
        self.example_combo['values'] = ('Clean Email Example', 'Spam Email Example')
        self.example_combo.grid(row=0, column=1, padx=5, pady=5)
        self.example_combo.bind('<<ComboboxSelected>>', self.load_example)
        
        # Email input
        ttk.Label(main_frame, text="Enter email content:").grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.email_text = scrolledtext.ScrolledText(main_frame, width=50, height=10)
        self.email_text.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Check button
        ttk.Button(main_frame, text="Check Spam", command=self.check_spam).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Result label
        self.result_label = ttk.Label(
            main_frame, 
            text="", 
            font=('Arial', 12, 'bold'),
            wraplength=400
        )
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)
    
    def check_spam(self):
        email_content = self.email_text.get("1.0", tk.END).strip()
        if email_content:
            try:
                # Transform text to vector
                email_vector = vectorizer.transform([email_content])
                # Make prediction
                prediction = model.predict(email_vector)
                probability = model.predict_proba(email_vector)[0]
                
                # Find most important words
                feature_importance = pd.DataFrame({
                    'word': vectorizer.get_feature_names_out(),
                    'importance': model.feature_importances_
                })
                top_words = feature_importance.nlargest(5, 'importance')
                
                if prediction[0] == 1:
                    result_text = f"This email might be SPAM!\nProbability of being spam: {probability[1]:.2%}\n\n"
                    result_text += "Important spam indicators:\n"
                    for _, row in top_words.iterrows():
                        if row['importance'] > 0:
                            result_text += f"- {row['word']}: {row['importance']:.4f}\n"
                    self.result_label.configure(foreground='red')
                else:
                    result_text = f"This email appears to be safe.\nProbability of being safe: {probability[0]:.2%}"
                    self.result_label.configure(foreground='green')
                
                self.result_label.configure(text=result_text)
                
            except Exception as e:
                self.result_label.configure(
                    text=f"Error occurred: {str(e)}", 
                    foreground='red'
                )
        else:
            self.result_label.configure(
                text="Please enter an email text!", 
                foreground='red'
            )

    def load_example(self, event=None):
        if self.example_var.get() == 'Clean Email Example':
            sample_text = """
            Hello,
            
            Please find attached the report I prepared for tomorrow's meeting.
            See you in the meeting room at 2:00 PM.
            
            Best regards
            """
        else:
            sample_text = """
            CONGRATULATIONS! YOU'VE WON $1,000,000!!!
            
            CLICK HERE NOW to claim your PRIZE: www.fakeprize.com
            LIMITED TIME OFFER - Don't miss this AMAZING opportunity!
            
            ACT NOW!!! URGENT!!!
            """
        
        self.email_text.delete("1.0", tk.END)
        self.email_text.insert("1.0", sample_text)

# Create main window and start application
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()
