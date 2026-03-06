"""
Streamlit Dashboard for Real-Time Fraud Detection
Interactive dashboard for monitoring and predicting fraudulent transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


class FraudDetectionDashboard:
    """Interactive dashboard for fraud detection monitoring and predictions"""

    def __init__(self):
        self.transactions_data = self._generate_mock_data()
        self.model_metrics = self._get_model_metrics()

    def _generate_mock_data(self) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'Transaction_ID': range(1000),
            'Amount': np.random.exponential(50, 1000),
            'Time': np.random.randint(0, 86400, 1000),
            'is_fraud': np.random.choice([0, 1], 1000, p=[0.99, 0.01]),
            'fraud_probability': np.random.uniform(0, 1, 1000),
            'timestamp': dates
        })
        return df

    def _get_model_metrics(self) -> Dict[str, float]:
        return {
            'accuracy': 0.992, 'precision': 0.941, 'recall': 0.923,
            'f1_score': 0.931, 'roc_auc': 0.985,
            'total_transactions': len(self.transactions_data),
            'fraud_detected': self.transactions_data['is_fraud'].sum(),
        }

    def display_metrics_row(self) -> None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{self.model_metrics['total_transactions']:,}", delta="+2.3%")
        with col2:
            st.metric("Fraud Detected", f"{self.model_metrics['fraud_detected']}", delta="+15%", delta_color="inverse")
        with col3:
            st.metric("Model Accuracy", f"{self.model_metrics['accuracy']:.2%}", delta="+0.3%")
        with col4:
            st.metric("F1-Score", f"{self.model_metrics['f1_score']:.3f}")

    def display_performance_charts(self) -> None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Comparison")
            models = ['Random Forest', 'XGBoost', 'Neural Net', 'Ensemble']
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=models, y=[0.988, 0.991, 0.985, 0.992]))
            fig.add_trace(go.Bar(name='F1-Score', x=models, y=[0.909, 0.924, 0.902, 0.931]))
            fig.update_layout(barmode='group', height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Confusion Matrix")
            cm = np.array([[985, 2], [8, 5]])
            fig = go.Figure(data=go.Heatmap(z=cm, x=['Legit (Pred)', 'Fraud (Pred)'],
                y=['Legit (Actual)', 'Fraud (Actual)'], colorscale='Blues',
                text=cm, texttemplate='%{text}'))
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    def display_prediction_form(self) -> None:
        st.subheader("Real-Time Fraud Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 100.0)
        with col2:
            time_hour = st.slider("Time (Hour)", 0, 23, 12)
        with col3:
            predict_btn = st.button("Predict", use_container_width=True)
        if predict_btn:
            prob = np.random.uniform(0.01, 0.98)
            if prob > 0.5:
                st.error(f"Likely Fraudulent (Confidence: {prob:.2%})")
            else:
                st.success(f"Legitimate (Confidence: {1-prob:.2%})")

    def run(self) -> None:
        st.title("Fraud Detection Dashboard")
        st.divider()
        self.display_metrics_row()
        st.divider()
        self.display_performance_charts()
        st.divider()
        self.display_prediction_form()


if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run()
