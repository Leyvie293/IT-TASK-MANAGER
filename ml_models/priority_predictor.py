"""
Priority Predictor Model
Predicts appropriate priority level for new tasks based on content and context
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class PriorityPredictor:
    """Predicts task priority based on content analysis"""
    
    def __init__(self, model_path='ml_models/priority_predictor.pkl'):
        self.model_path = model_path
        self.text_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.keyword_vectorizer = CountVectorizer(vocabulary=self._get_priority_keywords())
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_trained = False
        self.priority_levels = ['Low', 'Medium', 'High', 'Critical']
        
    def _get_priority_keywords(self):
        """Get keywords associated with different priority levels"""
        keywords = {
            'urgent': ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'now'],
            'high_impact': ['down', 'broken', 'not working', 'failed', 'error', 'issue', 'problem'],
            'time_sensitive': ['deadline', 'due', 'meeting', 'presentation', 'client', 'customer'],
            'business_critical': ['production', 'server', 'database', 'website', 'sales', 'revenue'],
            'security': ['security', 'breach', 'hack', 'attack', 'virus', 'malware', 'firewall'],
            'access': ['cannot work', 'no access', 'locked out', 'password', 'login'],
            'multiple_users': ['everyone', 'all users', 'team', 'department', 'company']
        }
        
        # Flatten keywords
        all_keywords = []
        for category_words in keywords.values():
            all_keywords.extend(category_words)
        
        return all_keywords
    
    def extract_features(self, title, description, context=None):
        """
        Extract features for priority prediction
        
        Args:
            title: Task title
            description: Task description
            context: Optional context dictionary
            
        Returns:
            Feature vector
        """
        # Combine text
        text = f"{title} {description}".lower()
        
        # Text features
        features = {}
        
        # 1. Text length features
        features['title_length'] = len(title.split())
        features['description_length'] = len(description.split()) if description else 0
        features['total_length'] = features['title_length'] + features['description_length']
        
        # 2. Keyword features
        urgency_keywords = ['urgent', 'emergency', 'critical', 'asap', 'immediately']
        error_keywords = ['error', 'failed', 'broken', 'not working', 'issue', 'problem']
        business_keywords = ['production', 'server', 'client', 'customer', 'revenue', 'sales']
        security_keywords = ['security', 'breach', 'hack', 'attack', 'virus']
        
        features['urgency_keyword_count'] = sum(1 for word in urgency_keywords if word in text)
        features['error_keyword_count'] = sum(1 for word in error_keywords if word in text)
        features['business_keyword_count'] = sum(1 for word in business_keywords if word in text)
        features['security_keyword_count'] = sum(1 for word in security_keywords if word in text)
        
        # 3. Capitalization features (often indicates urgency)
        features['caps_ratio_title'] = sum(1 for c in title if c.isupper()) / max(len(title), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # 4. Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_neg'] = sentiment['neg']
        features['sentiment_neu'] = sentiment['neu']
        features['sentiment_pos'] = sentiment['pos']
        features['sentiment_compound'] = sentiment['compound']
        
        # 5. Time features (if context provided)
        if context:
            created_hour = context.get('created_hour', 12)
            features['is_business_hours'] = 1 if 9 <= created_hour <= 17 else 0
            features['is_weekend'] = context.get('is_weekend', 0)
        else:
            features['is_business_hours'] = 1
            features['is_weekend'] = 0
        
        # 6. Department impact
        department = context.get('department', 'General') if context else 'General'
        high_impact_depts = ['IT', 'Security', 'Operations', 'Production']
        features['high_impact_department'] = 1 if department in high_impact_depts else 0
        
        # 7. User role impact
        user_role = context.get('user_role', 'Employee') if context else 'Employee'
        high_priority_roles = ['Manager', 'Director', 'Executive', 'Admin']
        features['high_priority_requester'] = 1 if user_role in high_priority_roles else 0
        
        # 8. Historical frequency (simulated)
        features['similar_tasks_recently'] = context.get('similar_tasks_count', 0) if context else 0
        
        return features
    
    def train(self, historical_data):
        """
        Train the priority predictor
        
        Args:
            historical_data: List of tasks with 'title', 'description', 'priority', and optional context
        """
        print("Training Priority Predictor...")
        
        df = pd.DataFrame(historical_data)
        
        # Prepare features
        feature_list = []
        labels = []
        
        for _, row in df.iterrows():
            features = self.extract_features(
                row['title'],
                row.get('description', ''),
                row.get('context', {})
            )
            feature_list.append(features)
            labels.append(row['priority'])
        
        # Create feature DataFrame
        features_df = pd.DataFrame(feature_list)
        
        # Encode labels
        self.label_encoder.fit(self.priority_levels)
        y = self.label_encoder.transform(labels)
        
        # Train classifier
        X = features_df.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.is_trained = True
        self.save_model()
        
        return accuracy
    
    def predict(self, title, description, context=None):
        """
        Predict priority for a task
        
        Args:
            title: Task title
            description: Task description
            context: Optional context dictionary
            
        Returns:
            Dictionary with priority prediction and details
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Train or load model first.")
        
        # Extract features
        features = self.extract_features(title, description, context)
        features_df = pd.DataFrame([features])
        
        # Predict
        X = features_df.values
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Get priority level
        priority = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence scores
        confidence_scores = {}
        for i, level in enumerate(self.label_encoder.classes_):
            confidence_scores[level] = float(probabilities[i])
        
        # Determine confidence level
        confidence = float(probabilities[prediction])
        confidence_level = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'
        
        # Get key reasons for prediction
        reasons = self._explain_prediction(features, priority)
        
        # Suggest escalation if needed
        suggested_action = 'Accept as predicted'
        if priority == 'Low' and features.get('urgency_keyword_count', 0) > 0:
            suggested_action = 'Review for potential escalation to Medium'
        elif priority == 'Medium' and features.get('security_keyword_count', 0) > 0:
            suggested_action = 'Consider escalation to High due to security implications'
        elif priority == 'High' and confidence < 0.6:
            suggested_action = 'Review with supervisor before assignment'
        
        return {
            'predicted_priority': priority,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'all_priorities': confidence_scores,
            'reasons': reasons,
            'suggested_action': suggested_action,
            'features': features,
            'probability_distribution': confidence_scores
        }
    
    def _explain_prediction(self, features, predicted_priority):
        """Generate human-readable explanation for prediction"""
        reasons = []
        
        # Check urgency keywords
        urgency_count = features.get('urgency_keyword_count', 0)
        if urgency_count > 0:
            reasons.append(f"Contains {urgency_count} urgency keyword(s)")
        
        # Check error keywords
        error_count = features.get('error_keyword_count', 0)
        if error_count > 0:
            reasons.append(f"Contains {error_count} error/problem keyword(s)")
        
        # Check security keywords
        security_count = features.get('security_keyword_count', 0)
        if security_count > 0:
            reasons.append(f"Contains {security_count} security-related keyword(s)")
        
        # Check sentiment
        sentiment = features.get('sentiment_compound', 0)
        if sentiment < -0.5:
            reasons.append("Strong negative sentiment detected")
        
        # Check department impact
        if features.get('high_impact_department', 0):
            reasons.append("High-impact department")
        
        # Check requester role
        if features.get('high_priority_requester', 0):
            reasons.append("High-priority requester")
        
        # Check time sensitivity
        if not features.get('is_business_hours', 1):
            reasons.append("Submitted outside business hours")
        
        # If no specific reasons found, provide generic explanation
        if not reasons:
            if predicted_priority == 'Critical':
                reasons.append("Multiple high-risk indicators detected")
            elif predicted_priority == 'High':
                reasons.append("Significant impact indicators present")
            elif predicted_priority == 'Medium':
                reasons.append("Standard business impact expected")
            else:
                reasons.append("Minimal impact indicators detected")
        
        return reasons
    
    def predict_batch(self, tasks):
        """
        Predict priorities for multiple tasks
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for task in tasks:
            result = self.predict(
                task['title'],
                task.get('description', ''),
                task.get('context', {})
            )
            results.append(result)
        return results
    
    def save_model(self):
        """Save model to disk"""
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'sentiment_analyzer': self.sentiment_analyzer,
            'is_trained': self.is_trained,
            'priority_levels': self.priority_levels
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        import os
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.sentiment_analyzer = model_data['sentiment_analyzer']
            self.is_trained = model_data['is_trained']
            self.priority_levels = model_data['priority_levels']
            
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
        from models.task_models import Task
        from models.user_models import User
        
        # Query tasks with priorities
        tasks = db_session.query(Task).filter(
            Task.priority.isnot(None),
            Task.status == 'Closed'
        ).limit(limit).all()
        
        training_data = []
        
        for task in tasks:
            # Get requester information
            requester = db_session.query(User).get(task.created_by) if task.created_by else None
            
            # Create context
            context = {
                'created_hour': task.created_at.hour if task.created_at else 12,
                'is_weekend': 1 if task.created_at and task.created_at.weekday() >= 5 else 0,
                'department': task.department or 'General',
                'user_role': requester.role if requester else 'Employee'
            }
            
            # Count similar recent tasks
            similar_tasks = db_session.query(Task).filter(
                Task.category == task.category,
                Task.created_at >= task.created_at - timedelta(days=7) if task.created_at else None,
                Task.id != task.id
            ).count()
            context['similar_tasks_count'] = similar_tasks
            
            training_data.append({
                'title': task.title,
                'description': task.description or '',
                'priority': task.priority,
                'context': context
            })
        
        return training_data
    
    def validate_priority_assignment(self, task, assigned_priority, confidence_threshold=0.6):
        """
        Validate if assigned priority is reasonable
        
        Args:
            task: Task dictionary
            assigned_priority: Currently assigned priority
            confidence_threshold: Minimum confidence for validation
            
        Returns:
            Validation result
        """
        prediction = self.predict(
            task['title'],
            task.get('description', ''),
            task.get('context', {})
        )
        
        predicted_priority = prediction['predicted_priority']
        confidence = prediction['confidence']
        
        validation = {
            'assigned_priority': assigned_priority,
            'predicted_priority': predicted_priority,
            'confidence': confidence,
            'match': assigned_priority == predicted_priority,
            'suggestion': None,
            'needs_review': False
        }
        
        # Check if validation is needed
        if not validation['match']:
            if confidence > confidence_threshold:
                validation['suggestion'] = f"Consider changing to {predicted_priority} (confidence: {confidence:.1%})"
                validation['needs_review'] = True
            else:
                validation['suggestion'] = "Review recommended but confidence is low"
                validation['needs_review'] = confidence > 0.4
        
        return validation

# Test the predictor
if __name__ == "__main__":
    # Create sample training data
    np.random.seed(42)
    
    # Sample data with realistic patterns
    sample_data = []
    
    # Critical tasks (security, production down)
    for i in range(50):
        sample_data.append({
            'title': f'URGENT: Production Server {i} DOWN',
            'description': f'Production server {i} completely unresponsive. All services affected. NEED IMMEDIATE ATTENTION!',
            'priority': 'Critical',
            'context': {
                'department': 'IT',
                'user_role': 'Manager',
                'created_hour': np.random.randint(0, 24),
                'is_weekend': np.random.choice([0, 1], p=[0.8, 0.2])
            }
        })
    
    # High priority tasks (important but not critical)
    for i in range(100):
        sample_data.append({
            'title': f'Email server issue {i}',
            'description': f'Email server experiencing performance issues. Some users cannot send emails.',
            'priority': 'High',
            'context': {
                'department': 'IT',
                'user_role': 'Employee',
                'created_hour': np.random.randint(9, 17),
                'is_weekend': 0
            }
        })
    
    # Medium priority tasks (standard issues)
    for i in range(200):
        sample_data.append({
            'title': f'Software installation request {i}',
            'description': f'Need Adobe Acrobat installed on workstation for document processing.',
            'priority': 'Medium',
            'context': {
                'department': np.random.choice(['HR', 'Finance', 'Marketing']),
                'user_role': 'Employee',
                'created_hour': np.random.randint(8, 18),
                'is_weekend': 0
            }
        })
    
    # Low priority tasks (minor issues)
    for i in range(150):
        sample_data.append({
            'title': f'Monitor adjustment {i}',
            'description': f'Monitor brightness needs adjustment for better viewing comfort.',
            'priority': 'Low',
            'context': {
                'department': np.random.choice(['Admin', 'Sales', 'Support']),
                'user_role': 'Employee',
                'created_hour': np.random.randint(8, 18),
                'is_weekend': 0
            }
        })
    
    # Create and train predictor
    predictor = PriorityPredictor()
    
    if not predictor.load_model():
        predictor.train(sample_data)
    
    # Test predictions
    test_cases = [
        {
            'title': 'SECURITY ALERT: Unauthorized access detected',
            'description': 'Multiple failed login attempts detected on admin portal. Possible breach attempt.',
            'context': {
                'department': 'Security',
                'user_role': 'Admin',
                'created_hour': 3,
                'is_weekend': 1
            }
        },
        {
            'title': 'Printer not working in finance department',
            'description': 'The main printer in finance is showing paper jam error. Need assistance.',
            'context': {
                'department': 'Finance',
                'user_role': 'Employee',
                'created_hour': 10,
                'is_weekend': 0
            }
        },
        {
            'title': 'New software license request',
            'description': 'Requesting license for statistical software for data analysis project.',
            'context': {
                'department': 'Research',
                'user_role': 'Analyst',
                'created_hour': 14,
                'is_weekend': 0
            }
        },
        {
            'title': 'URGENT: VPN not working for remote team',
            'description': 'Entire remote team cannot connect to VPN. Work completely stopped!',
            'context': {
                'department': 'IT',
                'user_role': 'Manager',
                'created_hour': 9,
                'is_weekend': 0
            }
        }
    ]
    
    print("\nPriority Prediction Test Results:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['title']}")
        print("-" * 60)
        
        result = predictor.predict(
            test_case['title'],
            test_case['description'],
            test_case['context']
        )
        
        print(f"Predicted Priority: {result['predicted_priority']}")
        print(f"Confidence: {result['confidence']:.1%} ({result['confidence_level']})")
        print(f"Suggested Action: {result['suggested_action']}")
        
        print("\nReasons:")
        for reason in result['reasons']:
            print(f"  • {reason}")
        
        print("\nProbability Distribution:")
        for priority, prob in result['all_priorities'].items():
            print(f"  {priority}: {prob:.1%}")