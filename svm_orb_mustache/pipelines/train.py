"""
SVM training module with hyperparameter search and model persistence.
Supports both Linear and RBF kernels with cross-validation.
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import joblib
import logging
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class SVMTrainer:
    """
    SVM trainer with hyperparameter optimization and evaluation.
    """
    
    def __init__(self, 
                 kernel: str = 'linear',
                 random_state: int = 42):
        """
        Initialize SVM trainer.
        
        Args:
            kernel: 'linear' or 'rbf'
            random_state: Random seed
        """
        self.kernel = kernel
        self.random_state = random_state
        self.svm = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_fitted = False
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray = None,
             y_val: np.ndarray = None,
             param_grid: Dict = None,
             cv_folds: int = 5) -> Dict:
        """
        Train SVM with optional hyperparameter search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            param_grid: Grid search parameters
            cv_folds: Number of CV folds
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training SVM with {self.kernel} kernel...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Setup parameter grid
        if param_grid is None:
            if self.kernel == 'linear':
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 100.0]
                }
            else:  # rbf
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
        
        # Create base estimator
        if self.kernel == 'linear':
            base_svm = LinearSVC(
                random_state=self.random_state,
                max_iter=10000,
                dual=False
            )
        else:
            base_svm = SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True,
                max_iter=10000
            )
        
        # Grid search
        logger.info(f"Performing grid search with {cv_folds}-fold CV...")
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        self.svm = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_fitted = True
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Evaluate on training set
        y_train_pred = self.svm.predict(X_train_scaled)
        train_metrics = self._compute_metrics(y_train, y_train_pred)
        
        logger.info(f"Training metrics: {train_metrics}")
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_pred = self.svm.predict(X_val_scaled)
            val_metrics = self._compute_metrics(y_val, y_val_pred)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return {
            'best_params': self.best_params,
            'best_cv_f1': float(grid_search.best_score_),
            'train': train_metrics,
            'val': val_metrics
        }
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                output_dir: str = None) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save plots
        
        Returns:
            Test metrics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating on test set...")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.svm.predict(X_test_scaled)
        
        metrics = self._compute_metrics(y_test, y_pred)
        
        # Get decision function or probability scores
        if hasattr(self.svm, 'decision_function'):
            y_scores = self.svm.decision_function(X_test_scaled)
        else:
            y_scores = self.svm.predict_proba(X_test_scaled)[:, 1]
        
        # Compute ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
            metrics['roc_auc'] = float(roc_auc)
            logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
        except:
            pass
        
        # Generate plots if output directory provided
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            self._plot_confusion_matrix(
                y_test, y_pred, 
                save_path=f"{output_dir}/confusion_matrix.png"
            )
            
            # PR curve
            self._plot_pr_curve(
                y_test, y_scores,
                save_path=f"{output_dir}/pr_curve.png"
            )
            
            # ROC curve
            self._plot_roc_curve(
                y_test, y_scores,
                save_path=f"{output_dir}/roc_curve.png"
            )
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def _compute_metrics(self, y_true, y_pred) -> Dict:
        """Compute classification metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0))
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_path: str):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Non-Face', 'Face']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_pr_curve(self, y_true, y_scores, save_path: str):
        """Plot and save precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {save_path}")
    
    def _plot_roc_curve(self, y_true, y_scores, save_path: str):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
    
    def save(self, svm_path: str, scaler_path: str):
        """Save model and scaler."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        joblib.dump(self.svm, svm_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {svm_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load(self, svm_path: str, scaler_path: str):
        """Load model and scaler."""
        self.svm = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        
        logger.info(f"Model loaded from {svm_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities or decision scores."""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.svm, 'predict_proba'):
            return self.svm.predict_proba(X_scaled)[:, 1]
        elif hasattr(self.svm, 'decision_function'):
            return self.svm.decision_function(X_scaled)
        else:
            return self.svm.predict(X_scaled).astype(float)
