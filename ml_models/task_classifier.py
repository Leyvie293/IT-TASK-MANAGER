"""
Task Classifier Model
Automatically classifies incoming tasks based on description and title
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from datetime import datetime
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """Text preprocessing for task descriptions"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.it_keywords = {
            'network': ['wifi', 'internet', 'vpn', 'firewall', 'router', 'switch', 'dns', 'ip', 'lan', 'wan'],
            'hardware': ['computer', 'laptop', 'printer', 'scanner', 'monitor', 'keyboard', 'mouse', 'cpu', 'ram'],
            'software': ['windows', 'office', 'outlook', 'excel', 'word', 'install', 'update', 'license'],
            'security': ['camera', 'cctv', 'alarm', 'access', 'biometric', 'surveillance', 'lock', 'keycard'],
            'email': ['outlook', 'gmail', 'mail', 'spam', 'inbox', 'attachment', 'send', 'receive'],
            'erp': ['sap', 'oracle', 'dynamics', 'erp', 'system', 'module', 'report'],
            'access': ['login', 'password', 'account', 'permission', 'access', 'privilege'],
            'backup': ['backup', 'restore', 'recovery', 'data', 'storage', 'drive']
        }
        
    def preprocess(self, text):
        """Preprocess text for classification"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract IT-specific features from text"""
        text_lower = text.lower()
        features = {}
        
        # Check for IT keywords
        for category, keywords in self.it_keywords.items():
            features[f'has_{category}'] = any(keyword in text_lower for keyword in keywords)
        
        # Text length features
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        
        # Urgency indicators
        urgency_words = ['urgent', 'critical', 'emergency', 'asap', 'immediately', 'important']
        features['has_urgency'] = any(word in text_lower for word in urgency_words)
        
        # Error indicators
        error_words = ['error', 'failed', 'broken', 'not working', 'issue', 'problem']
        features['has_error'] = any(word in text_lower for word in error_words)
        
        return features

class TaskClassifier:
    """Machine Learning model for task classification"""
    
    def __init__(self, model_path='ml_models/task_classifier.pkl'):
        self.model_path = model_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.categories = []
        
    def train(self, tasks_data):
        """
        Train the classifier on historical task data
        
        Args:
            tasks_data: List of dictionaries with 'title', 'description', 'category'
        """
        print("Training Task Classifier...")
        
        # Prepare data
        df = pd.DataFrame(tasks_data)
        
        # Combine title and description
        df['text'] = df['title'] + ' ' + df['description']
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        # Extract features
        features_list = []
        for text in df['text']:
            features = self.preprocessor.extract_features(text)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Encode labels
        self.categories = sorted(df['category'].unique())
        self.label_encoder.fit(self.categories)
        y = self.label_encoder.transform(df['category'])
        
        # Prepare text features
        X_text = self.vectorizer.fit_transform(df['processed_text'])
        
        # Combine features
        X_features = features_df.values
        X = np.hstack([X_text.toarray(), X_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        self.is_trained = True
        self.save_model()
        
        return accuracy
    
    def predict(self, title, description):
        """
        Predict task category based on title and description
        
        Args:
            title: Task title
            description: Task description
            
        Returns:
            Dictionary with predicted category and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Train or load model first.")
        
        # Combine text
        text = f"{title} {description}"
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Extract features
        features = self.preprocessor.extract_features(text)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([processed_text]).toarray()
        
        # Combine features
        X = np.hstack([text_vector, features_array])
        
        # Predict
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Get category
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores for all categories
        confidence_scores = {}
        for i, cat in enumerate(self.label_encoder.classes_):
            confidence_scores[cat] = float(probabilities[i])
        
        return {
            'predicted_category': category,
            'confidence': float(probabilities[prediction]),
            'all_categories': confidence_scores,
            'suggested_category': category if confidence_scores[category] > 0.5 else 'General',
            'features': features
        }
    
    def predict_multiple(self, tasks):
        """
        Predict categories for multiple tasks
        
        Args:
            tasks: List of dictionaries with 'title' and 'description'
            
        Returns:
            List of prediction results
        """
        results = []
        for task in tasks:
            result = self.predict(task['title'], task['description'])
            results.append(result)
        return results
    
    def save_model(self):
        """Save the trained model to disk"""
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'categories': self.categories,
            'preprocessor': self.preprocessor,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.categories = model_data['categories']
            self.preprocessor = model_data['preprocessor']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No model found at {self.model_path}")
            return False
    
    def retrain_incremental(self, new_tasks):
        """
        Retrain model incrementally with new data
        
        Args:
            new_tasks: List of new tasks with 'title', 'description', 'category'
        """
        if not self.is_trained:
            return self.train(new_tasks)
        
        # Load current training data
        # In production, you would load from database
        # For now, we'll retrain from scratch with combined data
        print("Retraining model incrementally...")
        
        # This would be replaced with database query in production
        existing_data = []  # Load from database
        all_data = existing_data + new_tasks
        
        return self.train(all_data)
    
    def get_feature_importance(self):
        """Get feature importance for model interpretation"""
        if not self.is_trained:
            return {}
        
        # Get feature names
        feature_names = list(self.vectorizer.get_feature_names_out())
        feature_names += list(self.preprocessor.extract_features("").keys())
        
        # Get importance scores
        importances = self.classifier.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        importance_dict = {}
        for i in indices[:20]:  # Top 20 features
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(importances[i])
        
        return importance_dict
    
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
        
        # Query tasks with categories
        tasks = db_session.query(
            Task.title, Task.description, Task.category
        ).filter(
            Task.category.isnot(None),
            Task.status == 'Closed'
        ).limit(limit).all()
        
        training_data = []
        for task in tasks:
            training_data.append({
                'title': task.title,
                'description': task.description or '',
                'category': task.category
            })
        
        return training_data

# Test the classifier
if __name__ == "__main__":
    # Example training data
    sample_data = [
        {
            'title': 'Email not working',
            'description': 'Cannot send or receive emails in Outlook. Getting error message.',
            'category': 'Email'
        },
        {
            'title': 'Network connection issue',
            'description': 'No internet connection on desktop computer. WiFi shows connected but no access.',
            'category': 'Network'
        },
        {
            'title': 'Printer not printing',
            'description': 'HP LaserJet printer shows offline. Cannot print documents.',
            'category': 'Hardware'
        },
        {
            'title': 'Software installation needed',
            'description': 'Need Adobe Photoshop installed on workstation for graphic design work.',
            'category': 'Software'
        },
        {
            'title': 'Access card not working',
            'description': 'Employee access card not working on main door. Needs reprogramming.',
            'category': 'Access Control'
        }
    ]
    
    # Add more diverse examples
    for i in range(20):
        sample_data.append({
            'title': f'CCTV Camera {i} issue',
            'description': f'Security camera {i} showing black screen. Needs maintenance.',
            'category': 'Security'
        })
    
    for i in range(15):
        sample_data.append({
            'title': f'ERP Report {i} problem',
            'description': f'Cannot generate monthly report {i} in ERP system. Error occurs.',
            'category': 'ERP'
        })
    
    # Create and train classifier
    classifier = TaskClassifier()
    
    if not classifier.load_model():
        classifier.train(sample_data)
    
    # Test predictions
    test_cases = [
        ('WiFi not connecting', 'Cannot connect to office WiFi network. Shows authentication error.'),
        ('Outlook email problem', 'Email attachments not downloading in Outlook application.'),
        ('Biometric device faulty', 'Fingerprint scanner not recognizing employees at entrance.'),
        ('Server backup failed', 'Scheduled backup failed last night. Need to check logs.'),
        ('VPN connection issue', 'Cannot connect to company VPN from home. Connection times out.')
    ]
    
    print("\nTest Predictions:")
    print("-" * 80)
    for title, desc in test_cases:
        result = classifier.predict(title, desc)
        print(f"Title: {title}")
        print(f"Predicted: {result['predicted_category']} (Confidence: {result['confidence']:.2%})")
        print(f"Top 3 categories: {sorted(result['all_categories'].items(), key=lambda x: x[1], reverse=True)[:3]}")
        print("-" * 80)