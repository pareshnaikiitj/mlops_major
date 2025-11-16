from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import numpy as np

def main():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data        # shape (400,4096)
    y = data.target      # 400 labels (0-39)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # save model file named exactly as required
    dump({'model': clf, 'X_test': X_test, 'y_test': y_test}, 'savedmodel.pth')
    print("Training finished. savedmodel.pth created.")

if __name__ == "__main__":
    main()
