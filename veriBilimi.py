# Gerekli kütüphaneleri içe aktaralım
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

# Veri setini yükleyelim
print("Veri seti yükleniyor...")
df = pd.read_json("spamjson.json")

# Veri Önişleme
df = df[['v1', 'v2']]  # Gereksiz sütunları çıkar
df.columns = ['label', 'text']  # Sütun isimlerini düzenle
df['text'] = df['text'].astype(str)  # Metinleri string'e çevir
df['label'] = (df['label'] == 'spam').astype(int)  # Etiketleri 1/0 yap

# Önce length sütununu oluşturalım
df['length'] = df['text'].apply(len)

# Veri Analizi ve Görselleştirme
plt.figure(figsize=(15, 10))

# 1. Spam/Ham Dağılımı (Pasta Grafik)
plt.subplot(2, 2, 1)
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Normal', 'Spam'])
plt.title('Spam/Normal E-posta Dağılımı')

# 2. Metin Uzunluğu Dağılımı
plt.subplot(2, 2, 2)
df['text_length'] = df['text'].apply(len)
sns.histplot(data=df, x='text_length', hue='label', bins=50)
plt.title('E-posta Uzunluğu Dağılımı')
plt.xlabel('Metin Uzunluğu')
plt.ylabel('Frekans')

# 3. En Sık Kullanılan Kelimeler (Spam vs Normal)
def get_common_words(texts):
    words = []
    for text in texts:
        # Noktalama işaretlerini kaldır ve küçük harfe çevir
        text = re.sub(r'[^\w\s]', '', text.lower())
        words.extend(text.split())
    return Counter(words).most_common(10)

spam_words = get_common_words(df[df['label'] == 1]['text'])
normal_words = get_common_words(df[df['label'] == 0]['text'])

# Spam Kelimeler
plt.subplot(2, 2, 3)
words, counts = zip(*spam_words)
plt.barh(words, counts)
plt.title('En Sık Kullanılan Spam Kelimeleri')
plt.xlabel('Frekans')

# Normal Kelimeler
plt.subplot(2, 2, 4)
words, counts = zip(*normal_words)
plt.barh(words, counts)
plt.title('En Sık Kullanılan Normal Kelimeler')
plt.xlabel('Frekans')

plt.tight_layout()
plt.show()

# Stop words listesini alalım
def get_top_words(texts, n=10):
    # Tüm metinleri birleştir ve küçük harfe çevir
    words = ' '.join(texts).lower()
    
    # Noktalama işaretlerini kaldır
    words = re.sub(r'[^\w\s]', '', words)
    
    # Kelimelere ayır
    words = words.split()
    
    # Stop words'leri ve sayıları kaldır
    words = [word for word in words 
            if word not in ENGLISH_STOP_WORDS  # Stop words'leri kaldır
            and not word.isdigit()  # Sayıları kaldır
            and len(word) > 2]  # 2 karakterden kısa kelimeleri kaldır
    
    return pd.Series(words).value_counts().head(n)

# En sık kullanılan kelimeleri görselleştir
plt.figure(figsize=(15, 6))

# Spam e-postalardaki en sık kelimeler
plt.subplot(1, 2, 1)
spam_words = get_top_words(df[df['label'] == 1]['text'])
spam_words.plot(kind='bar', color='red', alpha=0.6)
plt.title('Spam E-postalardaki En Sık Kelimeler\n(Stop Words Hariç)')
plt.xlabel('Kelimeler')
plt.ylabel('Frekans')
plt.xticks(rotation=45, ha='right')

# Normal e-postalardaki en sık kelimeler
plt.subplot(1, 2, 2)
normal_words = get_top_words(df[df['label'] == 0]['text'])
normal_words.plot(kind='bar', color='blue', alpha=0.6)
plt.title('Normal E-postalardaki En Sık Kelimeler\n(Stop Words Hariç)')
plt.xlabel('Kelimeler')
plt.ylabel('Frekans')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# En sık kullanılan kelimeleri yazdır
print("\nSpam e-postalardaki en sık 10 kelime:")
print(spam_words)
print("\nNormal e-postalardaki en sık 10 kelime:")
print(normal_words)

# Model Eğitimi
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

# Model Performans Görselleştirmesi
y_pred = model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# En Önemli Kelimeler
feature_importance = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
})
top_features = feature_importance.nlargest(20, 'importance')

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y='word')
plt.title('En Önemli 20 Kelime')
plt.xlabel('Önem Derecesi')
plt.show()

# İstatistiksel özet ve model performans metrikleri
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# 1. İstatistiksel Özet
print("\nVeri Seti İstatistiksel Özeti:")
stats_df = pd.DataFrame({
    'Metrik': [
        'Toplam E-posta Sayısı',
        'Spam E-posta Sayısı',
        'Normal E-posta Sayısı',
        'Spam Oranı (%)',
        'Ortalama E-posta Uzunluğu',
        'Medyan E-posta Uzunluğu',
        'En Kısa E-posta',
        'En Uzun E-posta'
    ],
    'Değer': [
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

# 2. Model Performans Metrikleri Görselleştirmesi
# Classification Report'u DataFrame'e çevir
cr = classification_report(y_test, y_pred, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()

# Classification Report Görselleştirmesi
plt.figure(figsize=(10, 6))
sns.heatmap(cr_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.3f')
plt.title('Model Performans Metrikleri')
plt.show()

# 3. Precision-Recall Eğrisi
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, 'b-', label='Precision-Recall Eğrisi')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Eğrisi')
plt.grid(True)
plt.legend()
plt.show()

# 4. Feature Importance ile Kelime Analizi
# En önemli 20 kelime ve önem dereceleri
feature_importance = pd.DataFrame({
    'Kelime': vectorizer.get_feature_names_out(),
    'Önem': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Önem', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Önem', y='Kelime')
plt.title('Spam Tespitinde En Önemli 20 Kelime')
plt.xlabel('Önem Derecesi')
plt.show()

# 5. Confusion Matrix'i daha detaylı göster
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

# Confusion Matrix üzerine yüzdeleri ekle
cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j+0.5, i+0.7, f'({cm_percentages[i,j]:.1%})', 
                ha='center', va='center')

plt.show()

# 6. Model Skor Dağılımı
plt.figure(figsize=(10, 6))
scores = model.predict_proba(X_test)[:, 1]
sns.histplot(data=pd.DataFrame({
    'Skor': scores,
    'Gerçek Değer': y_test
}), x='Skor', hue='Gerçek Değer', bins=50)
plt.title('Model Tahmin Skorlarının Dağılımı')
plt.xlabel('Spam Olma Olasılığı')
plt.ylabel('Frekans')
plt.show()

# Veri setindeki sütunları ve TF-IDF özelliklerini birleştir
print(f"Veri setindeki toplam sütun sayısı: {3 + X.shape[1]}")

# İlk birkaç TF-IDF özelliğini göster
print("\nTF-IDF ile oluşturulan ilk 10 özellik:")
print(vectorizer.get_feature_names_out()[:10])

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam E-posta Dedektörü")
        
        # Ana frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Örnek e-postalar için Combobox
        ttk.Label(main_frame, text="Örnek e-postalar:").grid(row=0, column=0, sticky=tk.W)
        self.example_var = tk.StringVar()
        self.example_combo = ttk.Combobox(main_frame, textvariable=self.example_var)
        self.example_combo['values'] = ('Temiz E-posta Örneği', 'Spam E-posta Örneği')
        self.example_combo.grid(row=0, column=1, padx=5, pady=5)
        self.example_combo.bind('<<ComboboxSelected>>', self.load_example)
        
        # E-posta girişi
        ttk.Label(main_frame, text="E-posta içeriğini giriniz:").grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.email_text = scrolledtext.ScrolledText(main_frame, width=50, height=10)
        self.email_text.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Kontrol butonu
        ttk.Button(main_frame, text="Spam Kontrolü", command=self.check_spam).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Sonuç etiketi
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
                # Metni vektöre dönüştür
                email_vector = vectorizer.transform([email_content])
                # Tahmin yap
                prediction = model.predict(email_vector)
                probability = model.predict_proba(email_vector)[0]
                
                # En önemli kelimeleri bul
                feature_importance = pd.DataFrame({
                    'word': vectorizer.get_feature_names_out(),
                    'importance': model.feature_importances_
                })
                top_words = feature_importance.nlargest(5, 'importance')
                
                if prediction[0] == 1:
                    result_text = f"Bu e-posta SPAM olabilir!\nSpam olma ihtimali: {probability[1]:.2%}\n\n"
                    result_text += "Önemli spam belirteçleri:\n"
                    for _, row in top_words.iterrows():
                        if row['importance'] > 0:
                            result_text += f"- {row['word']}: {row['importance']:.4f}\n"
                    self.result_label.configure(foreground='red')
                else:
                    result_text = f"Bu e-posta güvenli görünüyor.\nGüvenli olma ihtimali: {probability[0]:.2%}"
                    self.result_label.configure(foreground='green')
                
                self.result_label.configure(text=result_text)
                
            except Exception as e:
                self.result_label.configure(
                    text=f"Hata oluştu: {str(e)}", 
                    foreground='red'
                )
        else:
            self.result_label.configure(
                text="Lütfen bir e-posta metni girin!", 
                foreground='red'
            )

    def load_example(self, event=None):
        if self.example_var.get() == 'Temiz E-posta Örneği':
            sample_text = """
            Merhaba,
            
            Yarınki toplantı için hazırladığım raporu ekte bulabilirsiniz.
            Saat 14:00'te toplantı salonunda görüşmek üzere.
            
            İyi çalışmalar
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

# Ana pencereyi oluştur ve uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()