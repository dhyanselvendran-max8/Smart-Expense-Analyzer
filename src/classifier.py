"""Transaction categorisation model and synthetic dataset generator."""

import random
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.text_cleaner import clean_description


def generate_synthetic_training_data(seed: int = config.RANDOM_SEED) -> Tuple[List[str], List[str]]:
    """Generate 960 synthetic labeled transaction descriptions (120 per category).

    Uses 10 merchant seeds per category with random payment-channel noise,
    reference numbers, and city names. Fully deterministic using specified random seed.
    """
    rng = random.Random(seed)
    descriptions: List[str] = []
    labels: List[str] = []

    templates = [
        "{merchant} {channel} REF{ref}",
        "{channel}/{ref}/{merchant}/{city}",
        "{merchant} {city} {channel}",
        "UPI/{ref}/{merchant}",
        "{channel} PAYMENT TO {merchant} {city}",
        "POS {ref} {merchant} {city}",
        "{merchant} STORE {city} REF{ref}",
        "{channel} TRF {merchant} REF{ref}",
        "{merchant} ONLINE {channel}",
        "DEBIT {channel} {merchant} {city}",
        "{merchant}",
        "{channel} {merchant}",
    ]

    for category in config.CATEGORIES:
        merchants = config.SEED_MERCHANTS[category]
        for _ in range(config.SAMPLES_PER_CATEGORY):
            merchant = rng.choice(merchants)
            channel = rng.choice(config.PAYMENT_CHANNELS)
            city = rng.choice(config.CITIES)
            ref_num = rng.randint(1000000, 99999999)  # 7-8 digit reference number
            tmpl = rng.choice(templates)
            desc = tmpl.format(
                merchant=merchant,
                channel=channel,
                city=city,
                ref=ref_num,
            )
            descriptions.append(desc)
            labels.append(category)

    return descriptions, labels


def build_pipeline() -> Pipeline:
    """Construct the TF-IDF FeatureUnion and LogisticRegression pipeline.

    - Word TF-IDF: unigrams + bigrams, sublinear_tf=True, clean_description preprocessor
    - Char TF-IDF: char_wb, 3-5 n-grams, sublinear_tf=True, clean_description preprocessor
    - Combined using FeatureUnion
    - LogisticRegression(C=8.0, max_iter=1000)
    """
    word_vectorizer = TfidfVectorizer(
        preprocessor=clean_description,
        ngram_range=config.WORD_TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        preprocessor=clean_description,
        analyzer="char_wb",
        ngram_range=config.CHAR_TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )

    features = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )

    clf = LogisticRegression(
        C=config.LOGISTIC_REGRESSION_C,
        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        random_state=config.RANDOM_SEED,
    )

    return Pipeline([("features", features), ("clf", clf)])


class TransactionClassifier:
    """Classifier wrapper providing model training, caching, and inference."""

    def __init__(self, model: Pipeline = None):
        self.model = model

    def train_default_model(self) -> "TransactionClassifier":
        """Train classifier on the deterministic synthetic starter dataset."""
        descriptions, labels = generate_synthetic_training_data()
        self.model = build_pipeline()
        self.model.fit(descriptions, labels)
        return self

    def predict_single(self, text: str) -> Tuple[str, float]:
        """Predict category and confidence for a single description string.

        Returns (category, confidence). If confidence < CONFIDENCE_THRESHOLD,
        category is returned as 'Uncategorised'.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        cleaned = clean_description(text)
        if not cleaned:
            return config.UNCATEGORISED_LABEL, 0.0

        probas = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_
        best_idx = int(np.argmax(probas))
        confidence = float(probas[best_idx])
        conf_rounded = round(confidence, 3)

        if conf_rounded < config.CONFIDENCE_THRESHOLD:
            return config.UNCATEGORISED_LABEL, conf_rounded
        return str(classes[best_idx]), conf_rounded

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorise all transactions in DataFrame.

        Adds:
        - 'category': Predicted category name or 'Uncategorised'
        - 'confidence': Highest probability rounded to 3 decimal places
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        if df.empty:
            df_out = df.copy()
            df_out["category"] = pd.Series(dtype=str)
            df_out["confidence"] = pd.Series(dtype=float)
            return df_out

        descriptions = df["description"].astype(str).tolist()
        probas = self.model.predict_proba(descriptions)
        classes = self.model.classes_

        best_indices = np.argmax(probas, axis=1)
        max_probas = np.max(probas, axis=1)

        categories = []
        confidences = []

        for idx, best_idx in enumerate(best_indices):
            conf = round(float(max_probas[idx]), 3)
            # Empty or stripped descriptions
            if not clean_description(descriptions[idx]):
                categories.append(config.UNCATEGORISED_LABEL)
                confidences.append(0.0)
            elif conf < config.CONFIDENCE_THRESHOLD:
                categories.append(config.UNCATEGORISED_LABEL)
                confidences.append(conf)
            else:
                categories.append(str(classes[best_idx]))
                confidences.append(conf)

        df_out = df.copy()
        df_out["category"] = categories
        df_out["confidence"] = confidences
        return df_out


def get_trained_classifier() -> TransactionClassifier:
    """Factory helper to obtain a trained TransactionClassifier."""
    classifier = TransactionClassifier()
    classifier.train_default_model()
    return classifier

