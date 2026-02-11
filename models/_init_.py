"""
Database Models for Riley Falcon I.T Task Manager
"""

from .database import db, BaseModel
from .user_models import User
from .task_models import Task, TaskActivity, TaskComment, Escalation, WorkflowTemplate, WorkflowStep, Report, TaskHistory, Attachment, TaskTemplate, KnowledgeBase

# Import all models for alembic migrations
__all__ = [
    'db', 
    'BaseModel', 
    'User', 
    'Task', 
    'TaskHistory',  # Added
    'TaskActivity', 
    'TaskComment', 
    'Escalation', 
    'WorkflowTemplate', 
    'WorkflowStep',  # Added (was missing)
    'Report',
    'Attachment',    # Added
    'TaskTemplate',  # Added
    'KnowledgeBase'  # Added
]

# Create a function to initialize the database
def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    return db