"""
ML Service - Integrates all ML models with the main application
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from ml_models.task_classifier import TaskClassifier
from ml_models.sla_predictor import SLAPredictor
from ml_models.technician_matcher import TechnicianMatcher
from ml_models.priority_predictor import PriorityPredictor

class MLService:
    """Main service for integrating ML models with the application"""
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        
        # Initialize models
        self.task_classifier = TaskClassifier()
        self.sla_predictor = SLAPredictor()
        self.technician_matcher = TechnicianMatcher()
        self.priority_predictor = PriorityPredictor()
        
        # Load or train models
        self._initialize_models()
        
        # Model performance tracking
        self.performance_metrics = {}
        self.last_training_date = None
        
    def _initialize_models(self):
        """Initialize all ML models"""
        print("Initializing ML Models...")
        
        # Try to load existing models
        models_loaded = 0
        
        if self.task_classifier.load_model():
            models_loaded += 1
            print("✓ Task Classifier loaded")
        else:
            print("✗ Task Classifier needs training")
        
        if self.sla_predictor.load_model():
            models_loaded += 1
            print("✓ SLA Predictor loaded")
        else:
            print("✗ SLA Predictor needs training")
        
        if self.technician_matcher.load_model():
            models_loaded += 1
            print("✓ Technician Matcher loaded")
        else:
            print("✗ Technician Matcher needs training")
        
        if self.priority_predictor.load_model():
            models_loaded += 1
            print("✓ Priority Predictor loaded")
        else:
            print("✗ Priority Predictor needs training")
        
        print(f"\n{models_loaded}/4 models loaded successfully")
        
        # If models are not loaded, they need to be trained with database data
        if models_loaded < 4 and self.db_session:
            self.train_all_models()
    
    def train_all_models(self):
        """Train all ML models with database data"""
        if not self.db_session:
            print("Database session not available for training")
            return False
        
        print("\n" + "="*60)
        print("TRAINING ALL ML MODELS")
        print("="*60)
        
        try:
            # 1. Train Task Classifier
            print("\n1. Training Task Classifier...")
            training_data = self.task_classifier.generate_training_data_from_database(
                self.db_session, limit=1000
            )
            if training_data:
                accuracy = self.task_classifier.train(training_data)
                self.performance_metrics['task_classifier'] = {
                    'accuracy': accuracy,
                    'trained_on': len(training_data),
                    'last_trained': datetime.utcnow().isoformat()
                }
            else:
                print("   Insufficient training data for Task Classifier")
            
            # 2. Train SLA Predictor
            print("\n2. Training SLA Predictor...")
            training_data = self.sla_predictor.generate_training_data_from_database(
                self.db_session, limit=1000
            )
            if training_data:
                metrics = self.sla_predictor.train(training_data)
                self.performance_metrics['sla_predictor'] = {
                    **metrics,
                    'trained_on': len(training_data),
                    'last_trained': datetime.utcnow().isoformat()
                }
            else:
                print("   Insufficient training data for SLA Predictor")
            
            # 3. Train Technician Matcher
            print("\n3. Training Technician Matcher...")
            technician_data = self.technician_matcher.generate_technician_data_from_database(
                self.db_session
            )
            if technician_data:
                # Create dummy historical assignments for initial training
                historical_assignments = []
                self.technician_matcher.train(historical_assignments, technician_data)
                self.performance_metrics['technician_matcher'] = {
                    'trained_on': len(technician_data),
                    'last_trained': datetime.utcnow().isoformat()
                }
            else:
                print("   Insufficient technician data for Technician Matcher")
            
            # 4. Train Priority Predictor
            print("\n4. Training Priority Predictor...")
            training_data = self.priority_predictor.generate_training_data_from_database(
                self.db_session, limit=1000
            )
            if training_data:
                accuracy = self.priority_predictor.train(training_data)
                self.performance_metrics['priority_predictor'] = {
                    'accuracy': accuracy,
                    'trained_on': len(training_data),
                    'last_trained': datetime.utcnow().isoformat()
                }
            else:
                print("   Insufficient training data for Priority Predictor")
            
            self.last_training_date = datetime.utcnow()
            
            print("\n" + "="*60)
            print("MODEL TRAINING COMPLETE")
            print("="*60)
            
            # Save performance metrics
            self._save_performance_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_new_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a new task using all ML models
        
        Args:
            task_data: Dictionary with task information
            
        Returns:
            Comprehensive analysis results
        """
        analysis = {
            'task_id': task_data.get('id'),
            'timestamp': datetime.utcnow().isoformat(),
            'models_used': []
        }
        
        # 1. Predict Category
        if self.task_classifier.is_trained:
            try:
                category_result = self.task_classifier.predict(
                    task_data.get('title', ''),
                    task_data.get('description', '')
                )
                analysis['category_prediction'] = category_result
                analysis['models_used'].append('task_classifier')
            except Exception as e:
                analysis['category_prediction_error'] = str(e)
        
        # 2. Predict Priority
        if self.priority_predictor.is_trained:
            try:
                context = {
                    'department': task_data.get('department'),
                    'created_hour': datetime.utcnow().hour,
                    'is_weekend': 1 if datetime.utcnow().weekday() >= 5 else 0
                }
                
                priority_result = self.priority_predictor.predict(
                    task_data.get('title', ''),
                    task_data.get('description', ''),
                    context
                )
                analysis['priority_prediction'] = priority_result
                analysis['models_used'].append('priority_predictor')
            except Exception as e:
                analysis['priority_prediction_error'] = str(e)
        
        # 3. Predict SLA and Resolution Time
        if self.sla_predictor.is_trained:
            try:
                # Create task features for SLA prediction
                task_features = {
                    'title': task_data.get('title', ''),
                    'description': task_data.get('description', ''),
                    'category': analysis.get('category_prediction', {}).get('predicted_category', 'General'),
                    'priority': analysis.get('priority_prediction', {}).get('predicted_priority', 'Medium'),
                    'department': task_data.get('department', 'General'),
                    'complexity': 'Medium'  # Default, could be refined
                }
                
                sla_result = self.sla_predictor.predict(task_features)
                analysis['sla_prediction'] = sla_result
                analysis['models_used'].append('sla_predictor')
            except Exception as e:
                analysis['sla_prediction_error'] = str(e)
        
        # 4. Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # 5. Confidence score (average of all model confidences)
        confidences = []
        if 'category_prediction' in analysis:
            confidences.append(analysis['category_prediction'].get('confidence', 0))
        if 'priority_prediction' in analysis:
            confidences.append(analysis['priority_prediction'].get('confidence', 0))
        if 'sla_prediction' in analysis:
            confidences.append(analysis['sla_prediction'].get('confidence', 0))
        
        analysis['overall_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        return analysis
    
    def find_best_technician(self, task_data: Dict[str, Any], available_technicians: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find best technician for a task
        
        Args:
            task_data: Task information
            available_technicians: List of available technicians
            
        Returns:
            Matching results
        """
        if not self.technician_matcher.is_trained:
            return {'error': 'Technician matcher not trained'}
        
        try:
            # Update technician profiles
            self.technician_matcher.update_technician_profiles(available_technicians)
            
            # Find best matches
            matches = self.technician_matcher.find_best_match(
                task_data,
                [tech['id'] for tech in available_technicians],
                top_n=3
            )
            
            return {
                'matches': matches,
                'total_technicians': len(available_technicians),
                'recommended_count': len(matches)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def batch_analyze_tasks(self, tasks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple tasks
        
        Args:
            tasks_data: List of task dictionaries
            
        Returns:
            List of analysis results
        """
        results = []
        for task_data in tasks_data:
            analysis = self.analyze_new_task(task_data)
            results.append(analysis)
        
        # Generate batch insights
        batch_insights = self._generate_batch_insights(results)
        
        return {
            'individual_analyses': results,
            'batch_insights': batch_insights
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from analysis"""
        recommendations = []
        
        # Category recommendations
        if 'category_prediction' in analysis:
            cat_result = analysis['category_prediction']
            if cat_result.get('confidence', 0) < 0.6:
                recommendations.append(
                    f"Category prediction confidence is low ({cat_result['confidence']:.1%}). "
                    "Please verify the category manually."
                )
            else:
                recommendations.append(
                    f"Suggested category: {cat_result['predicted_category']} "
                    f"(confidence: {cat_result['confidence']:.1%})"
                )
        
        # Priority recommendations
        if 'priority_prediction' in analysis:
            pri_result = analysis['priority_prediction']
            recommendations.append(
                f"Predicted priority: {pri_result['predicted_priority']} "
                f"(confidence: {pri_result['confidence']:.1%})"
            )
            
            if pri_result.get('suggested_action'):
                recommendations.append(pri_result['suggested_action'])
        
        # SLA recommendations
        if 'sla_prediction' in analysis:
            sla_result = analysis['sla_prediction']
            if sla_result.get('risk_level') in ['High', 'Critical']:
                recommendations.append(
                    f"High SLA risk detected: {sla_result['risk_level']}. "
                    "Consider assigning to experienced technician and monitoring closely."
                )
            
            if sla_result.get('recommendations'):
                recommendations.extend(sla_result['recommendations'])
        
        # General recommendations
        if analysis.get('overall_confidence', 0) < 0.5:
            recommendations.append(
                "Overall prediction confidence is low. Manual review recommended."
            )
        
        return recommendations
    
    def _generate_batch_insights(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from batch analysis"""
        if not analyses:
            return {}
        
        insights = {
            'total_tasks': len(analyses),
            'priority_distribution': {},
            'category_distribution': {},
            'risk_assessment': {},
            'confidence_summary': {}
        }
        
        # Collect data
        priorities = []
        categories = []
        risks = []
        confidences = []
        
        for analysis in analyses:
            # Priority
            if 'priority_prediction' in analysis:
                priority = analysis['priority_prediction'].get('predicted_priority')
                if priority:
                    priorities.append(priority)
            
            # Category
            if 'category_prediction' in analysis:
                category = analysis['category_prediction'].get('predicted_category')
                if category:
                    categories.append(category)
            
            # Risk
            if 'sla_prediction' in analysis:
                risk = analysis['sla_prediction'].get('risk_level')
                if risk:
                    risks.append(risk)
            
            # Confidence
            confidences.append(analysis.get('overall_confidence', 0))
        
        # Calculate distributions
        if priorities:
            from collections import Counter
            insights['priority_distribution'] = dict(Counter(priorities))
        
        if categories:
            from collections import Counter
            insights['category_distribution'] = dict(Counter(categories))
        
        if risks:
            from collections import Counter
            insights['risk_assessment'] = dict(Counter(risks))
        
        if confidences:
            insights['confidence_summary'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'low_confidence_tasks': len([c for c in confidences if c < 0.5])
            }
        
        # Generate actionable insights
        actionable_insights = []
        
        # Check for high-risk tasks
        high_risk_count = insights['risk_assessment'].get('High', 0) + insights['risk_assessment'].get('Critical', 0)
        if high_risk_count > 0:
            actionable_insights.append(
                f"{high_risk_count} task(s) have high SLA risk. Prioritize these for assignment."
            )
        
        # Check for resource needs based on categories
        if insights['category_distribution']:
            most_common_category = max(insights['category_distribution'].items(), key=lambda x: x[1])
            actionable_insights.append(
                f"Most common category: {most_common_category[0]} ({most_common_category[1]} tasks). "
                "Ensure adequate technician coverage for this category."
            )
        
        # Check for priority distribution
        if insights['priority_distribution']:
            critical_tasks = insights['priority_distribution'].get('Critical', 0)
            if critical_tasks > 0:
                actionable_insights.append(
                    f"{critical_tasks} critical task(s) detected. Immediate attention required."
                )
        
        insights['actionable_insights'] = actionable_insights
        
        return insights
    
    def _save_performance_metrics(self):
        """Save model performance metrics to file"""
        metrics_file = 'ml_models/performance_metrics.json'
        
        metrics_data = {
            'last_updated': datetime.utcnow().isoformat(),
            'models': self.performance_metrics,
            'training_date': self.last_training_date.isoformat() if self.last_training_date else None
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_performance_metrics(self):
        """Load model performance metrics"""
        metrics_file = 'ml_models/performance_metrics.json'
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            self.performance_metrics = metrics_data.get('models', {})
            if metrics_data.get('training_date'):
                self.last_training_date = datetime.fromisoformat(metrics_data['training_date'])
            
            return metrics_data
        
        return {}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models"""
        status = {
            'task_classifier': {
                'is_trained': self.task_classifier.is_trained,
                'categories': self.task_classifier.categories if self.task_classifier.is_trained else []
            },
            'sla_predictor': {
                'is_trained': self.sla_predictor.is_trained
            },
            'technician_matcher': {
                'is_trained': self.technician_matcher.is_trained,
                'technician_count': len(self.technician_matcher.technician_profiles)
            },
            'priority_predictor': {
                'is_trained': self.priority_predictor.is_trained,
                'priority_levels': self.priority_predictor.priority_levels
            },
            'last_training': self.last_training_date.isoformat() if self.last_training_date else None,
            'performance_metrics': self.performance_metrics
        }
        
        return status
    
    def retrain_models_if_needed(self, retrain_threshold_days: int = 30):
        """Retrain models if they haven't been trained recently"""
        if not self.last_training_date:
            print("Models have never been trained. Training now...")
            return self.train_all_models()
        
        days_since_training = (datetime.utcnow() - self.last_training_date).days
        
        if days_since_training >= retrain_threshold_days:
            print(f"Models were last trained {days_since_training} days ago. Retraining...")
            return self.train_all_models()
        else:
            print(f"Models were trained {days_since_training} days ago. No retraining needed.")
            return False

# Flask Blueprint for ML API endpoints
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required

ml_bp = Blueprint('ml', __name__, url_prefix='/api/ml')

@ml_bp.route('/status', methods=['GET'])
@login_required
def get_ml_status():
    """Get ML models status"""
    ml_service = current_app.ml_service
    status = ml_service.get_model_status()
    return jsonify(status)

@ml_bp.route('/analyze-task', methods=['POST'])
@login_required
def analyze_task():
    """Analyze a task using ML models"""
    data = request.json
    
    if not data or 'title' not in data:
        return jsonify({'error': 'Task title is required'}), 400
    
    ml_service = current_app.ml_service
    analysis = ml_service.analyze_new_task(data)
    
    return jsonify(analysis)

@ml_bp.route('/find-technician', methods=['POST'])
@login_required
def find_technician():
    """Find best technician for a task"""
    data = request.json
    
    if not data or 'task' not in data or 'technicians' not in data:
        return jsonify({'error': 'Task and technicians data required'}), 400
    
    ml_service = current_app.ml_service
    result = ml_service.find_best_technician(data['task'], data['technicians'])
    
    return jsonify(result)

@ml_bp.route('/train-models', methods=['POST'])
@login_required
def train_models():
    """Train all ML models"""
    ml_service = current_app.ml_service
    success = ml_service.train_all_models()
    
    if success:
        return jsonify({'success': True, 'message': 'Models trained successfully'})
    else:
        return jsonify({'success': False, 'message': 'Model training failed'}), 500

@ml_bp.route('/batch-analyze', methods=['POST'])
@login_required
def batch_analyze():
    """Analyze multiple tasks"""
    data = request.json
    
    if not data or 'tasks' not in data:
        return jsonify({'error': 'Tasks data required'}), 400
    
    ml_service = current_app.ml_service
    result = ml_service.batch_analyze_tasks(data['tasks'])
    
    return jsonify(result)

@ml_bp.route('/performance', methods=['GET'])
@login_required
def get_performance():
    """Get model performance metrics"""
    ml_service = current_app.ml_service
    metrics = ml_service.load_performance_metrics()
    
    return jsonify(metrics)

# Initialize ML Service in Flask app
def init_ml_service(app, db_session):
    """Initialize ML Service with Flask app"""
    ml_service = MLService(db_session)
    app.ml_service = ml_service
    
    # Register blueprint
    app.register_blueprint(ml_bp)
    
    # Schedule periodic retraining
    from apscheduler.schedulers.background import BackgroundScheduler
    
    scheduler = BackgroundScheduler()
    
    # Retrain models every 30 days
    @scheduler.scheduled_job('interval', days=30)
    def retrain_models_job():
        with app.app_context():
            ml_service.retrain_models_if_needed()
    
    scheduler.start()
    
    return ml_service