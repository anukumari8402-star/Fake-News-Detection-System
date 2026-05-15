import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

# ── Load dataset ──────────────────────────────────────────────
fake = pd.read_csv(r"e:\Fake.csv", encoding="ISO-8859-1")
true = pd.read_csv(r"e:\True (1).csv", encoding="ISO-8859-1")

# ── Drop existing label column if present ──────────────
for df in [fake, true]:
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)

fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data = data[['text', 'label']].dropna()
data = data[data['text'].str.strip() != ""]
data['label'] = pd.to_numeric(data['label'], errors='coerce')
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)
data.reset_index(drop=True, inplace=True)

print("Label distribution before balancing:")
print(data['label'].value_counts())

# ── Balance ───────────────────────────────────────────────────
fake_df = data[data.label == 0]
real_df = data[data.label == 1]
min_size = min(len(fake_df), len(real_df))
fake_balanced = resample(fake_df, replace=False, n_samples=min_size, random_state=42)
real_balanced = resample(real_df, replace=False, n_samples=min_size, random_state=42)
data = pd.concat([fake_balanced, real_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
print("\nAfter balancing:")
print(data['label'].value_counts())

# ── Consistent clean_text (same as app.py) ─────────────
def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text).strip()   # ← was missing before
    return text

data['text'] = data['text'].apply(clean_text)

# Remove rows that became empty after cleaning
data = data[data['text'].str.len() >= 20].reset_index(drop=True)

# ── Split ─────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.25, random_state=42
)

# ── TF-IDF ───────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True
)
xv_train = vectorizer.fit_transform(x_train)
xv_test  = vectorizer.transform(x_test)

# ── Only use models that support predict_proba ─────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        class_weight='balanced'
    ),
    "Naive Bayes": MultinomialNB(alpha=0.1)
}

best_accuracy = 0
best_model    = None
best_name     = ""

for name, model in models.items():
    model.fit(xv_train, y_train)
    pred = model.predict(xv_test)
    acc  = accuracy_score(y_test, pred)

    print(f"\n{name} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, pred)}")

    # ── skip models without predict_proba ─────────────
    if not hasattr(model, "predict_proba"):
        print(f"  ⚠️  {name} has no predict_proba — skipped for selection")
        continue

    if acc > best_accuracy:
        best_accuracy = acc
        best_model    = model
        best_name     = name

print(f"\n✅ Best Model : {best_name}")
print(f"   Accuracy   : {best_accuracy:.4f}")

# ── Sanity check: both classes should be predicted ────────────
final_preds = best_model.predict(xv_test)
print("\nFinal prediction distribution:")
print(pd.Series(final_preds).value_counts())

# ── Save ──────────────────────────────────────────────────────
pickle.dump(best_model,    open("model.pkl",      "wb"))
pickle.dump(vectorizer,    open("vectorizer.pkl", "wb"))
pickle.dump(best_accuracy, open("accuracy.pkl",   "wb"))
print("\nSaved model.pkl, vectorizer.pkl, accuracy.pkl ✅")