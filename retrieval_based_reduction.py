from tabicl import TabICLClassifier
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


seeds = 5

for seed in tqdm(range(seeds)):

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    clf = TabICLClassifier(n_estimators=1, random_state=seed, k = 50)
    clf.fit(X_train, y_train)  # this is cheap

    #res, rollout = clf.predict_with_rollout(X_test, return_row_emb=False)
    res = clf.predict(X_test)

    #Score
    acc = accuracy_score(y_test, res)
    prec = precision_score(y_test, res)
    rec = recall_score(y_test, res)
    f1 = f1_score(y_test, res)

    #Print
    print(f"Seed {seed}: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")