import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import re
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


traind = pd.read_csv("train_data.csv")
testd = pd.read_csv("test_data.csv")
test_sample = testd["sample"]

lemmatizare = WordNetLemmatizer()
stop_words = set(stopwords.words('romanian'))

def preprocesare_dialect(text):
    text = text.lower()
    text = re.sub(r'\$NE\$', '[NE]', text)
    text = re.sub(r'\b\d+-\d+\b', '', text)
    text = re.sub(r'[^\w\săâîșțĂÂÎȘȚ]', '', text)
    return text


def preprocesare_categorii(text, category = None):
    text = text.lower()
    text = re.sub(r'\$NE\$', '[NE]', text)
    text = re.sub(r'[^\w\săâîșțĂÂÎȘȚ]', '', text)
    text = re.sub(r'\b\d+-\d+\b', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizare.lemmatize(word) for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

traind['procesat_dialect'] = traind['sample'].apply(preprocesare_dialect)
testd['procesat_dialect'] = testd['sample'].apply(preprocesare_dialect)

traind['procesat_categorie'] = traind['sample'].apply(preprocesare_categorii)
testd['procesat_categorie'] = testd['sample'].apply(preprocesare_categorii)

cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=30000,
        min_df=0.02)),
    ('classifier', CalibratedClassifierCV(estimator=LinearSVC(max_iter=100000,
                                                              C=10,
                                                              class_weight='balanced'), cv=10))
])

modela_cat = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=50000,
        max_df = 0.2)),
    ('classifier', CalibratedClassifierCV(estimator=LinearSVC(max_iter=50000,
                                                              C = 5,
                                                              dual = True,
                                                              penalty ='l2'), cv=cross_val))
])

rf_model_cat = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        max_features=30000,
        max_df=0.9)),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))
])

base_learners_cat = [
    ('svm_cat', modela_cat),
    ('rf_cat', rf_model_cat)
]

model_cat = StackingClassifier(
    estimators=base_learners_cat,
    final_estimator=LogisticRegression(solver='lbfgs', max_iter=10000)
)

print("Antrenare model dialect...")
model.fit(traind["procesat_dialect"], traind["dialect"])

print("Antrenare model categorii...")
model_cat.fit(traind["procesat_categorie"], traind["category"])

predictie = model.predict(testd['procesat_dialect'])
predictie_cat = model_cat.predict(testd['procesat_categorie'])

rezultat = pd.DataFrame({
    'datapointId': testd['datapointID'],
    'dialect': predictie,
    'category': predictie_cat
})

rezultat.to_csv("output.csv", index=False)

#acuratete
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    traind['procesat_dialect'],
    traind['dialect'],
    test_size=0.2,
    random_state=42)

model.fit(X_train, y_train)
val_preds = model.predict(X_val)
print(classification_report(y_val, val_preds))

X_train, X_val, y_train, y_val = train_test_split(
    traind['procesat_categorie'],
    traind['category'],
    test_size=0.2,
    random_state=42)

model_cat.fit(X_train, y_train)
val_preds = model_cat.predict(X_val)
print("\nRaport performanță categorie:")
print(classification_report(y_val, val_preds))


print("Rezultatele au fost salvate în output.csv!")