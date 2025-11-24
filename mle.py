"""
Discrete Naive Bayes with Maximum Likelihood Estimation
for Student Performance dataset 
- 80/20 train-test split.
- Evaluates MSE and accuracy.

Run as a script or import the NaiveBayesMLE class in a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

class NaiveBayesMLE:
    """
    assume all features X_j and target Y are discrete 
    """

    def __init__(self, laplace_alpha: float = 1e-9):
        """
        laplace_alpha:
            - 0.0  -> pure MLE (can give zero probabilities)
            - >0.0 -> Laplace smoothing
        """
        self.laplace_alpha = laplace_alpha
        self.class_priors_ = None          # P（G3= xxx）shape [n_classes]
        self.feature_cond_probs_ = None    # list of arrays, one per feature- each array shape [n_class, n_values_of_feature]
        self.class_values_ = None          # store sorted unique class labels
        self.feature_value_maps_ = None    # list of arrays, each array- sorted unique values per feature

    def fit(self, X: np.ndarray, y: np.ndarray):
        #get: class prior probability table P（Y=c); All conditional probability tables that maximize likelihood# 
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Each column must be discrete integers.
        y : array-like of shape (n_samples,)
            Discrete class labels (integers).
        """
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)

        n_samples, n_features = X.shape

        # Unique class values & mapping to indices
        self.class_values_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.class_values_)

        # Prior P(Y = c)
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)

        if self.laplace_alpha == 0.0:
            self.class_priors_ = class_counts / n_samples
        else:
            self.class_priors_ = (class_counts + self.laplace_alpha) / (
                n_samples + self.laplace_alpha * n_classes
            )

        # For each feature, compute P(X_j = v | Y = c)
        self.feature_cond_probs_ = []
        self.feature_value_maps_ = []

        for j in range(n_features):
            xj = X[:, j]
            values_j, xj_idx = np.unique(xj, return_inverse=True)
            self.feature_value_maps_.append(values_j)
            n_values_j = len(values_j)

            # counts[c, v] = number of samples where y=c and X_j=value v
            counts = np.zeros((n_classes, n_values_j), dtype=float)

            for i in range(n_samples):
                c = y_idx[i]
                v = xj_idx[i]
                counts[c, v] += 1.0

            # convert to conditional probabilities with Laplace smoothing
            if self.laplace_alpha == 0.0:
                # pure MLE; may have zeros
                cond_probs = counts / class_counts[:, None]
            else:
                cond_probs = (counts + self.laplace_alpha) / (
                    class_counts[:, None] + self.laplace_alpha * n_values_j
                )

            self.feature_cond_probs_.append(cond_probs)

        return self

    def _log_proba_single(self, x_row: np.ndarray) -> np.ndarray:
        """
        Compute log P(Y = c, X = x_row) for one sample
        """
        x_row = np.asarray(x_row, dtype=int)
        n_features = len(self.feature_cond_probs_)
        log_priors = np.log(self.class_priors_ + 1e-30)  # avoid log(0)
        log_joint = log_priors.copy()

        for j in range(n_features):
            values_j = self.feature_value_maps_[j]
            cond_probs_j = self.feature_cond_probs_[j]

            # find index of x_row[j] within values_j
            # if unseen value appears, we assign a very small probability
            if x_row[j] in values_j:
                v_idx = np.where(values_j == x_row[j])[0][0]
                p_x_given_y = cond_probs_j[:, v_idx]
            else:
                # unseen value -> tiny probability
                p_x_given_y = np.full_like(self.class_priors_, 1e-12)

            log_joint += np.log(p_x_given_y + 1e-30)

        return log_joint

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely class for each sample.
        """
        X = np.asarray(X, dtype=int)
        n_samples = X.shape[0]
        preds_idx = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            log_joint = self._log_proba_single(X[i])
            preds_idx[i] = np.argmax(log_joint)

        # map back from index to actual class label
        return self.class_values_[preds_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities P(Y = c | X) using Bayes rule
        (up to a normalization constant).
        """
        X = np.asarray(X, dtype=int)
        n_samples = X.shape[0]
        n_classes = len(self.class_values_)
        probs = np.zeros((n_samples, n_classes), dtype=float)

        for i in range(n_samples):
            log_joint = self._log_proba_single(X[i])
            # convert log-joint to probabilities via softmax
            max_log = np.max(log_joint)
            exp_shifted = np.exp(log_joint - max_log)
            probs[i] = exp_shifted / exp_shifted.sum()

        return probs


def load_and_prepare_dataframe(csv_path: str,
                               target_col: str,
                               feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV and return (X_df, y_series).

    - If feature_cols is None, all columns except target_col are used as features.
    - Assumes all chosen columns are already encoded as integers.
    """
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Ensure integer dtype (categories already encoded)
    X = X.astype(int)
    y = y.astype(int)

    return X, y


def main():
    # 1. input directory
    csv_path = "external_factors_df.csv"  
    target_col = "G3_category" 
    feature_cols = None #let load_and_prepare_dataframe take all non-target columns.

   # 2. load dta 
    X_df, y = load_and_prepare_dataframe(
        csv_path=csv_path,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    # 3. TRAIN–TEST SPLIT (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y.values,
        test_size=0.2,
        random_state=42,
        stratify=y.values  # keep class distribution similar
    )

    # 4. FIT NAIVE BAYES MLE MODEL 
    model = NaiveBayesMLE(laplace_alpha=1e-6) 
    model.fit(X_train, y_train)

    # 5. PREDICT ON TEST SET
    y_pred = model.predict(X_test)

    # 6. EVALUATE
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("===== Naive Bayes MLE Results =====")
    print(f"Test MSE:      {mse:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Multi-class precision, recall, F1
    prec_macro  = precision_score(y_test, y_pred, average='macro')
    rec_macro   = recall_score(y_test, y_pred, average='macro')
    f1_macro    = f1_score(y_test, y_pred, average='macro')

    prec_weight = precision_score(y_test, y_pred, average='weighted')
    rec_weight  = recall_score(y_test, y_pred, average='weighted')
    f1_weight   = f1_score(y_test, y_pred, average='weighted')

    print("\n===== MACRO METRICS =====")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f}")

    print("\n===== WEIGHTED METRICS =====")
    print(f"Precision (weighted): {prec_weight:.4f}")
    print(f"Recall (weighted):    {rec_weight:.4f}")
    print(f"F1 (weighted):        {f1_weight:.4f}")

    # Per-class details (very useful)
    print("\n===== PER-CLASS REPORT =====")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
