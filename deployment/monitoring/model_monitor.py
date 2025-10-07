"""
Model Monitoring and Drift Detection
Tracks model performance and detects data/concept drift
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model predictions and detect drift"""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        """
        Initialize monitor with reference (training) data
        
        Args:
            reference_data: Training data for comparison
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data) if reference_data is not None else None
        self.prediction_log = []
        
    def _compute_stats(self, data: pd.DataFrame) -> Dict:
        """Compute statistical summary of data"""
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median())
                }
            else:
                stats[col] = {
                    'unique': int(data[col].nunique()),
                    'mode': str(data[col].mode()[0]) if len(data[col].mode()) > 0 else None,
                    'value_counts': data[col].value_counts().to_dict()
                }
        return stats
    
    def log_prediction(self, input_data: Dict, prediction: int, probability: float):
        """
        Log prediction for monitoring
        
        Args:
            input_data: Input features
            prediction: Model prediction
            probability: Prediction probability
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': int(prediction),
            'probability': float(probability),
        }
        self.prediction_log.append(log_entry)
        
    def detect_data_drift(self, new_data: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """
        Detect data drift using statistical tests
        
        Args:
            new_data: Recent prediction data
            threshold: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_stats is None:
            logger.warning("No reference data available for drift detection")
            return {'error': 'No reference data'}
        
        drift_detected = {}
        
        for col in new_data.columns:
            if col not in self.reference_data.columns:
                continue
                
            if new_data[col].dtype in ['int64', 'float64']:
                # Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    new_data[col].dropna()
                )
                
                drift_detected[col] = {
                    'test': 'Kolmogorov-Smirnov',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift': p_value < threshold,
                    'severity': 'high' if p_value < 0.01 else 'medium' if p_value < threshold else 'low'
                }
            else:
                # Chi-square test for categorical features
                ref_counts = self.reference_data[col].value_counts()
                new_counts = new_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                new_freq = [new_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                if sum(new_freq) > 0 and sum(ref_freq) > 0:
                    statistic, p_value = stats.chisquare(new_freq, ref_freq)
                    
                    drift_detected[col] = {
                        'test': 'Chi-Square',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift': p_value < threshold,
                        'severity': 'high' if p_value < 0.01 else 'medium' if p_value < threshold else 'low'
                    }
        
        # Summary
        drifted_features = [k for k, v in drift_detected.items() if v.get('drift', False)]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(drift_detected),
            'drifted_features': len(drifted_features),
            'drift_percentage': len(drifted_features) / len(drift_detected) * 100 if len(drift_detected) > 0 else 0,
            'features': drift_detected,
            'alert': len(drifted_features) > len(drift_detected) * 0.3  # Alert if >30% features drift
        }
    
    def detect_concept_drift(self, window_size: int = 100) -> Dict:
        """
        Detect concept drift by analyzing prediction patterns
        
        Args:
            window_size: Size of sliding window for analysis
            
        Returns:
            Dictionary with concept drift analysis
        """
        if len(self.prediction_log) < window_size * 2:
            return {'error': 'Insufficient data for concept drift detection'}
        
        # Split into old and recent windows
        recent_window = self.prediction_log[-window_size:]
        old_window = self.prediction_log[-window_size*2:-window_size]
        
        # Calculate metrics for each window
        recent_probs = [p['probability'] for p in recent_window]
        old_probs = [p['probability'] for p in old_window]
        
        recent_preds = [p['prediction'] for p in recent_window]
        old_preds = [p['prediction'] for p in old_window]
        
        # Statistical tests
        prob_statistic, prob_pvalue = stats.ks_2samp(old_probs, recent_probs)
        
        # Prediction distribution change
        recent_positive_rate = sum(recent_preds) / len(recent_preds)
        old_positive_rate = sum(old_preds) / len(old_preds)
        rate_change = abs(recent_positive_rate - old_positive_rate)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'window_size': window_size,
            'probability_drift': {
                'test': 'Kolmogorov-Smirnov',
                'statistic': float(prob_statistic),
                'p_value': float(prob_pvalue),
                'drift': prob_pvalue < 0.05
            },
            'prediction_rate_change': {
                'old_positive_rate': float(old_positive_rate),
                'recent_positive_rate': float(recent_positive_rate),
                'absolute_change': float(rate_change),
                'relative_change': float(rate_change / old_positive_rate) if old_positive_rate > 0 else 0,
                'significant': rate_change > 0.1
            },
            'alert': prob_pvalue < 0.05 or rate_change > 0.1
        }
    
    def get_performance_metrics(self, actual_labels: List[int] = None) -> Dict:
        """
        Calculate performance metrics if actual labels are available
        
        Args:
            actual_labels: True labels for recent predictions
            
        Returns:
            Performance metrics
        """
        if not self.prediction_log:
            return {'error': 'No predictions logged'}
        
        predictions = [p['prediction'] for p in self.prediction_log]
        probabilities = [p['probability'] for p in self.prediction_log]
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'positive_prediction_rate': sum(predictions) / len(predictions),
            'average_probability': np.mean(probabilities),
            'probability_std': np.std(probabilities)
        }
        
        if actual_labels and len(actual_labels) == len(predictions):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics.update({
                'accuracy': float(accuracy_score(actual_labels, predictions)),
                'precision': float(precision_score(actual_labels, predictions, zero_division=0)),
                'recall': float(recall_score(actual_labels, predictions, zero_division=0)),
                'f1_score': float(f1_score(actual_labels, predictions, zero_division=0)),
                'roc_auc': float(roc_auc_score(actual_labels, probabilities))
            })
        
        return metrics
    
    def export_logs(self, filepath: str):
        """Export prediction logs to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.prediction_log, f, indent=2)
        logger.info(f"Exported {len(self.prediction_log)} predictions to {filepath}")
    
    def generate_report(self, new_data: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive monitoring report
        
        Args:
            new_data: Recent input data for drift analysis
            
        Returns:
            Complete monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_period': {
                'total_predictions': len(self.prediction_log),
                'first_prediction': self.prediction_log[0]['timestamp'] if self.prediction_log else None,
                'last_prediction': self.prediction_log[-1]['timestamp'] if self.prediction_log else None
            }
        }
        
        # Add performance metrics
        report['performance'] = self.get_performance_metrics()
        
        # Add drift detection
        if new_data is not None:
            report['data_drift'] = self.detect_data_drift(new_data)
        
        if len(self.prediction_log) >= 200:
            report['concept_drift'] = self.detect_concept_drift()
        
        return report


# Example usage
if __name__ == "__main__":
    # Example monitoring workflow
    monitor = ModelMonitor()
    
    # Simulate predictions
    for i in range(150):
        monitor.log_prediction(
            input_data={'age': 30 + i % 20, 'duration': 100 + i % 50},
            prediction=1 if i % 3 == 0 else 0,
            probability=0.6 + (i % 40) / 100
        )
    
    # Generate report
    report = monitor.generate_report()
    print(json.dumps(report, indent=2))
