"""
SLA Predictor Model
Predicts SLA compliance and resolution time for tasks
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SLAPredictor:
    """Predict SLA compliance and resolution time"""
    
    def __init__(self, model_path='ml_models/sla_predictor.pkl'):
        self.model_path = model_path
        self.resolution_time_model = None
        self.sla_compliance_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, task_data):
        """
        Prepare features for SLA prediction
        
        Args:
            task_data: Dictionary or DataFrame with task features
            
        Returns:
            numpy array of features
        """
        if isinstance(task_data, dict):
            task_data = pd.DataFrame([task_data])
        
        # Create feature DataFrame
        features = pd.DataFrame()
        
        # Categorical features
        categorical_features = ['category', 'priority', 'department', 'complexity']
        for feature in categorical_features:
            if feature in task_data.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    # Fit on all possible values
                    all_values = task_data[feature].unique()
                    self.label_encoders[feature].fit(all_values)
                features[feature] = self.label_encoders[feature].transform(task_data[feature])
        
        # Numerical features
        numerical_features = [
            'word_count_title', 'word_count_description',
            'has_attachments', 'has_urgency_words', 'time_of_day',
            'day_of_week', 'month', 'technician_experience',
            'current_workload', 'similar_tasks_completed'
        ]
        
        for feature in numerical_features:
            if feature in task_data.columns:
                features[feature] = task_data[feature].astype(float)
            else:
                # Set default values
                if feature == 'word_count_title':
                    features[feature] = task_data.get('title', '').apply(lambda x: len(str(x).split()))
                elif feature == 'word_count_description':
                    features[feature] = task_data.get('description', '').apply(lambda x: len(str(x).split()))
                else:
                    features[feature] = 0
        
        # One-hot encoding for missing categorical features
        for feature in categorical_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        # Reorder columns to match training
        if self.feature_columns:
            features = features[self.feature_columns]
        
        return features
    
    def extract_features_from_task(self, task, technician_data=None):
        """
        Extract features from a task object
        
        Args:
            task: Task object from database
            technician_data: Optional technician information
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic task features
        features['category'] = task.category or 'General'
        features['priority'] = task.priority or 'Medium'
        features['department'] = task.department or 'General'
        features['complexity'] = task.complexity or 'Medium'
        
        # Text features
        features['word_count_title'] = len(str(task.title).split())
        features['word_count_description'] = len(str(task.description).split())
        
        # Attachment features
        features['has_attachments'] = 1 if task.attachments and len(task.attachments) > 0 else 0
        
        # Urgency indicators
        urgency_words = ['urgent', 'critical', 'emergency', 'asap', 'immediately']
        text_lower = (str(task.title) + ' ' + str(task.description)).lower()
        features['has_urgency_words'] = any(word in text_lower for word in urgency_words)
        
        # Time features
        created_time = task.created_at or datetime.utcnow()
        features['time_of_day'] = created_time.hour
        features['day_of_week'] = created_time.weekday()
        features['month'] = created_time.month
        
        # Technician features
        if technician_data:
            features['technician_experience'] = technician_data.get('experience_months', 0)
            features['current_workload'] = technician_data.get('current_tasks', 0)
            features['similar_tasks_completed'] = technician_data.get('similar_tasks_completed', 0)
        else:
            features['technician_experience'] = 12  # Default 1 year
            features['current_workload'] = 3  # Default moderate workload
            features['similar_tasks_completed'] = 5  # Default
        
        return features
    
    def train(self, historical_data):
        """
        Train SLA prediction models
        
        Args:
            historical_data: List of completed tasks with resolution times
        """
        print("Training SLA Predictor...")
        
        df = pd.DataFrame(historical_data)
        
        # Prepare features
        features_list = []
        resolution_times = []
        sla_compliances = []
        
        for _, row in df.iterrows():
            features = {}
            
            # Categorical features
            for col in ['category', 'priority', 'department', 'complexity']:
                features[col] = row.get(col, 'Unknown')
            
            # Text features
            features['word_count_title'] = len(str(row.get('title', '')).split())
            features['word_count_description'] = len(str(row.get('description', '')).split())
            
            # Other features
            features['has_attachments'] = row.get('has_attachments', 0)
            features['has_urgency_words'] = row.get('has_urgency_words', 0)
            features['time_of_day'] = row.get('created_hour', 12)
            features['day_of_week'] = row.get('created_weekday', 0)
            features['month'] = row.get('created_month', 1)
            features['technician_experience'] = row.get('technician_experience', 12)
            features['current_workload'] = row.get('technician_workload', 3)
            features['similar_tasks_completed'] = row.get('similar_tasks_completed', 5)
            
            features_list.append(features)
            
            # Target: resolution time in hours
            resolution_time = row.get('resolution_time_hours', 24)
            resolution_times.append(resolution_time)
            
            # Target: SLA compliance (1 = met, 0 = breached)
            sla_compliance = 1 if row.get('sla_met', True) else 0
            sla_compliances.append(sla_compliance)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Encode categorical features
        categorical_cols = ['category', 'priority', 'department', 'complexity']
        for col in categorical_cols:
            if col in features_df.columns:
                self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col])
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        # Scale features
        X = self.scaler.fit_transform(features_df)
        y_resolution = np.array(resolution_times)
        y_compliance = np.array(sla_compliances)
        
        # Split data
        X_train, X_test, y_train_res, y_test_res = train_test_split(
            X, y_resolution, test_size=0.2, random_state=42
        )
        
        _, _, y_train_comp, y_test_comp = train_test_split(
            X, y_compliance, test_size=0.2, random_state=42
        )
        
        # Train resolution time model (XGBoost)
        self.resolution_time_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.resolution_time_model.fit(X_train, y_train_res)
        
        # Train SLA compliance model (Random Forest)
        self.sla_compliance_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.sla_compliance_model.fit(X_train, y_train_comp)
        
        # Evaluate models
        y_pred_res = self.resolution_time_model.predict(X_test)
        y_pred_comp = self.sla_compliance_model.predict(X_test)
        
        mae_res = mean_absolute_error(y_test_res, y_pred_res)
        mse_res = mean_squared_error(y_test_res, y_pred_res)
        r2_res = r2_score(y_test_res, y_pred_res)
        
        accuracy_comp = accuracy_score(y_test_comp, y_pred_comp)
        
        print(f"Resolution Time Model:")
        print(f"  MAE: {mae_res:.2f} hours")
        print(f"  MSE: {mse_res:.2f}")
        print(f"  R²: {r2_res:.2%}")
        
        print(f"\nSLA Compliance Model:")
        print(f"  Accuracy: {accuracy_comp:.2%}")
        
        self.is_trained = True
        self.save_model()
        
        return {
            'resolution_model_metrics': {'mae': mae_res, 'mse': mse_res, 'r2': r2_res},
            'compliance_model_accuracy': accuracy_comp
        }
    
    def predict(self, task_features, technician_features=None):
        """
        Predict resolution time and SLA compliance
        
        Args:
            task_features: Dictionary with task features
            technician_features: Optional dictionary with technician features
            
        Returns:
            Dictionary with predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Train or load model first.")
        
        # Combine features
        features = self.extract_features_from_task(task_features, technician_features)
        features_df = pd.DataFrame([features])
        
        # Encode categorical features
        for col in ['category', 'priority', 'department', 'complexity']:
            if col in features_df.columns and col in self.label_encoders:
                try:
                    features_df[col] = self.label_encoders[col].transform([features[col]])[0]
                except:
                    # Handle unseen labels
                    features_df[col] = 0
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_columns]
        
        # Scale features
        X = self.scaler.transform(features_df)
        
        # Predict
        predicted_time = self.resolution_time_model.predict(X)[0]
        compliance_prob = self.sla_compliance_model.predict_proba(X)[0]
        
        # Get SLA compliance prediction
        sla_met_probability = compliance_prob[1]  # Probability of meeting SLA
        sla_met = sla_met_probability > 0.5
        
        # Calculate suggested SLA based on priority
        priority = task_features.get('priority', 'Medium')
        sla_hours = {
            'Critical': 2,
            'High': 4,
            'Medium': 24,
            'Low': 72
        }
        
        suggested_sla = sla_hours.get(priority, 24)
        
        # Risk assessment
        risk_level = 'Low'
        if predicted_time > suggested_sla * 0.8:
            risk_level = 'Medium'
        if predicted_time > suggested_sla:
            risk_level = 'High'
        if sla_met_probability < 0.3:
            risk_level = 'Critical'
        
        return {
            'predicted_resolution_time_hours': float(predicted_time),
            'sla_met_probability': float(sla_met_probability),
            'sla_met_prediction': bool(sla_met),
            'suggested_sla_hours': suggested_sla,
            'risk_level': risk_level,
            'confidence': float(min(sla_met_probability, 1 - sla_met_probability) * 2),
            'recommendations': self._generate_recommendations(predicted_time, suggested_sla, risk_level)
        }
    
    def _generate_recommendations(self, predicted_time, suggested_sla, risk_level):
        """Generate recommendations based on predictions"""
        recommendations = []
        
        time_difference = predicted_time - suggested_sla
        
        if time_difference > 0:
            recommendations.append(
                f"Task is predicted to take {time_difference:.1f} hours longer than SLA. "
                f"Consider assigning to experienced technician."
            )
        
        if risk_level == 'High':
            recommendations.append(
                "High risk of SLA breach. Recommend immediate attention and escalation plan."
            )
        elif risk_level == 'Medium':
            recommendations.append(
                "Medium risk. Monitor closely and provide additional resources if needed."
            )
        
        if predicted_time > 8:
            recommendations.append(
                "Predicted resolution time exceeds one work day. Consider breaking down into subtasks."
            )
        
        return recommendations
    
    def predict_batch(self, tasks):
        """
        Predict for multiple tasks
        
        Args:
            tasks: List of task feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for task in tasks:
            result = self.predict(task)
            results.append(result)
        return results
    
    def save_model(self):
        """Save trained models to disk"""
        model_data = {
            'resolution_time_model': self.resolution_time_model,
            'sla_compliance_model': self.sla_compliance_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained models from disk"""
        import os
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.resolution_time_model = model_data['resolution_time_model']
            self.sla_compliance_model = model_data['sla_compliance_model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No model found at {self.model_path}")
            return False
    
    def generate_training_data_from_database(self, db_session, limit=1000):
        """
        Generate training data from database
        
        Args:
            db_session: SQLAlchemy session
            limit: Maximum number of tasks to use
            
        Returns:
            List of training data
        """
        from models.task_models import Task, User
        from datetime import datetime
        
        # Query completed tasks with resolution times
        tasks = db_session.query(Task).filter(
            Task.status == 'Closed',
            Task.start_time.isnot(None),
            Task.end_time.isnot(None)
        ).limit(limit).all()
        
        training_data = []
        
        for task in tasks:
            # Calculate resolution time
            if task.start_time and task.end_time:
                resolution_hours = (task.end_time - task.start_time).total_seconds() / 3600
            else:
                resolution_hours = 24  # Default
            
            # Check if SLA was met
            sla_met = True
            if task.sla_due_date and task.end_time:
                sla_met = task.end_time <= task.sla_due_date
            
            # Get technician data
            technician = None
            if task.assigned_to:
                technician = db_session.query(User).get(task.assigned_to)
            
            data = {
                'title': task.title,
                'description': task.description,
                'category': task.category,
                'priority': task.priority,
                'department': task.department,
                'complexity': task.complexity or 'Medium',
                'has_attachments': 1 if task.attachments and len(task.attachments) > 0 else 0,
                'created_hour': task.created_at.hour if task.created_at else 12,
                'created_weekday': task.created_at.weekday() if task.created_at else 0,
                'created_month': task.created_at.month if task.created_at else 1,
                'resolution_time_hours': resolution_hours,
                'sla_met': sla_met
            }
            
            # Add technician features if available
            if technician:
                data['technician_experience'] = technician.get('experience_months', 12)
                data['technician_workload'] = technician.get('current_tasks', 0)
                
                # Count similar tasks completed by technician
                similar_tasks = db_session.query(Task).filter(
                    Task.assigned_to == technician.id,
                    Task.category == task.category,
                    Task.status == 'Closed'
                ).count()
                data['similar_tasks_completed'] = similar_tasks
            
            # Check for urgency words
            text = (task.title + ' ' + (task.description or '')).lower()
            urgency_words = ['urgent', 'critical', 'emergency', 'asap', 'immediately']
            data['has_urgency_words'] = any(word in text for word in urgency_words)
            
            training_data.append(data)
        
        return training_data

# Test the predictor
if __name__ == "__main__":
    # Create sample training data
    np.random.seed(42)
    
    categories = ['Network', 'Hardware', 'Software', 'Email', 'Security']
    priorities = ['Critical', 'High', 'Medium', 'Low']
    departments = ['IT', 'HR', 'Finance', 'Operations']
    complexities = ['Simple', 'Medium', 'Complex']
    
    sample_data = []
    
    for i in range(200):
        category = np.random.choice(categories)
        priority = np.random.choice(priorities)
        
        # Base resolution time based on priority
        base_times = {'Critical': 4, 'High': 8, 'Medium': 24, 'Low': 48}
        base_time = base_times[priority]
        
        # Add randomness
        resolution_time = np.random.normal(base_time, base_time * 0.3)
        resolution_time = max(1, resolution_time)  # Minimum 1 hour
        
        # Determine if SLA was met (80% chance for medium/low, 60% for high/critical)
        sla_met_prob = 0.8 if priority in ['Medium', 'Low'] else 0.6
        sla_met = np.random.random() < sla_met_prob
        
        data = {
            'title': f'Sample Task {i}',
            'description': f'Description for task {i}',
            'category': category,
            'priority': priority,
            'department': np.random.choice(departments),
            'complexity': np.random.choice(complexities),
            'has_attachments': np.random.choice([0, 1]),
            'created_hour': np.random.randint(8, 18),
            'created_weekday': np.random.randint(0, 5),
            'created_month': np.random.randint(1, 13),
            'technician_experience': np.random.randint(6, 60),
            'technician_workload': np.random.randint(1, 10),
            'similar_tasks_completed': np.random.randint(0, 50),
            'has_urgency_words': np.random.choice([0, 1]),
            'resolution_time_hours': resolution_time,
            'sla_met': sla_met
        }
        
        sample_data.append(data)
    
    # Create and train predictor
    predictor = SLAPredictor()
    
    if not predictor.load_model():
        predictor.train(sample_data)
    
    # Test predictions
    test_task = {
        'title': 'Server down emergency',
        'description': 'Production server not responding. All services affected.',
        'category': 'Network',
        'priority': 'Critical',
        'department': 'IT',
        'complexity': 'Complex',
        'has_attachments': 0
    }
    
    test_technician = {
        'experience_months': 24,
        'current_tasks': 2,
        'similar_tasks_completed': 15
    }
    
    result = predictor.predict(test_task, test_technician)
    
    print("\nSLA Prediction Results:")
    print("=" * 60)
    print(f"Task: {test_task['title']}")
    print(f"Priority: {test_task['priority']}")
    print(f"Predicted Resolution Time: {result['predicted_resolution_time_hours']:.1f} hours")
    print(f"SLA Met Probability: {result['sla_met_probability']:.2%}")
    print(f"Suggested SLA: {result['suggested_sla_hours']} hours")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")