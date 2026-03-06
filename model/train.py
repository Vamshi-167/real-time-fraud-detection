"""
Model Training Pipeline for Fraud Detection
Trains ensemble models: Random Forest, XGBoost, Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import argparse
import logging
from typing import Dict, Tuple, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Training pipeline for fraud detection ensemble model"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate transaction data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} transactions with {df.shape[1]} features")
        logger.info(f"Fraud rate: {df["Class"].mean():.4%}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)

        X = df_processed.drop(["Class"], axis=1).values
        y = df_processed["Class"].values

        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Feature matrix shape: {X_scaled.shape}")
        return X_scaled, y

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf
        logger.info("Random Forest training complete")
        return rf

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost classifier"""
        logger.info("Training XGBoost...")
        scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state, use_label_encoder=False,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train, y_train)
        self.models["xgboost"] = xgb_model
        logger.info("XGBoost training complete")
        return xgb_model

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, name: str) -> Dict[str, float]:
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        self.metrics[name] = metrics
        logger.info(f"{name} - Accuracy: {metrics[\"accuracy\"]:.4f}, F1: {metrics[\"f1_score\"]:.4f}, AUC: {metrics[\"roc_auc\"]:.4f}")
        return metrics

    def ensemble_predict(self, X: np.ndarray, weights: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions with weighted averaging"""
        if weights is None:
            weights = {"random_forest": 0.3, "xgboost": 0.7}

        ensemble_probs = np.zeros(X.shape[0])
        for name, weight in weights.items():
            if name in self.models:
                probs = self.models[name].predict_proba(X)[:, 1]
                ensemble_probs += weight * probs

        predictions = (ensemble_probs >= 0.5).astype(int)
        return predictions, ensemble_probs

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict[str, float]:
        """Perform stratified k-fold cross-validation"""
        logger.info(f"Running {n_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        cv_scores = cross_val_score(
            self.models.get("xgboost", self.models["random_forest"]),
            X, y, cv=skf, scoring="f1"
        )

        results = {"mean_f1": cv_scores.mean(), "std_f1": cv_scores.std()}
        logger.info(f"CV F1: {results[\"mean_f1\"]:.4f} +/- {results[\"std_f1\"]:.4f}")
        return results

    def save_model(self, filepath: str) -> None:
        """Save trained models and scaler"""
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "metrics": self.metrics
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def train(self, data_path: str, output_path: str) -> Dict[str, Dict[str, float]]:
        """Execute the complete training pipeline"""
        df = self.load_data(data_path)
        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)

        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)

        ens_pred, ens_prob = self.ensemble_predict(X_test)
        ens_metrics = {
            "accuracy": accuracy_score(y_test, ens_pred),
            "precision": precision_score(y_test, ens_pred, zero_division=0),
            "recall": recall_score(y_test, ens_pred, zero_division=0),
            "f1_score": f1_score(y_test, ens_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, ens_prob)
        }
        self.metrics["ensemble"] = ens_metrics
        logger.info(f"Ensemble - Accuracy: {ens_metrics[\"accuracy\"]:.4f}, F1: {ens_metrics[\"f1_score\"]:.4f}")

        self.cross_validate(X, y)
        self.save_model(output_path)

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="models/fraud_model.pkl", help="Output path")
    args = parser.parse_args()

    trainer = FraudDetectionTrainer()
    metrics = trainer.train(args.data, args.output)

    print("\n=== Training Complete ===")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
