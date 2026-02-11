#!/usr/bin/env python3
"""
ML Model Training Script
Run this script to train all ML models for the I.T Task Manager
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models.database import db
from services.ml_service import MLService

def main():
    """Main training function"""
    print("="*70)
    print("RILEY FALCON I.T TASK MANAGER - ML MODEL TRAINING")
    print("="*70)
    
    # Create Flask app
    app = create_app()
    
    with app.app_context():
        # Initialize database session
        db_session = db.session
        
        # Create ML service
        ml_service = MLService(db_session)
        
        # Train all models
        print("\nStarting ML Model Training...")
        success = ml_service.train_all_models()
        
        if success:
            print("\n✅ All ML models trained successfully!")
            
            # Display model status
            status = ml_service.get_model_status()
            print("\nModel Status:")
            print("-"*40)
            
            for model_name, model_info in status.items():
                if isinstance(model_info, dict):
                    trained = "✅" if model_info.get('is_trained') else "❌"
                    print(f"{model_name}: {trained}")
                    
                    if model_name == 'task_classifier' and model_info.get('categories'):
                        print(f"  Categories: {len(model_info['categories'])}")
                    elif model_name == 'technician_matcher' and model_info.get('technician_count'):
                        print(f"  Technicians: {model_info['technician_count']}")
                    elif model_name == 'priority_predictor' and model_info.get('priority_levels'):
                        print(f"  Priorities: {len(model_info['priority_levels'])}")
            
            print("\nPerformance Metrics:")
            print("-"*40)
            for model_name, metrics in ml_service.performance_metrics.items():
                print(f"\n{model_name}:")
                for key, value in metrics.items():
                    if key != 'last_trained':
                        print(f"  {key}: {value}")
            
            print("\n📊 Training complete! Models are ready for use.")
            
        else:
            print("\n❌ Model training failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()