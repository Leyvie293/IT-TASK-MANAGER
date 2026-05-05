# models/database.py - COMPLETE FIXED VERSION - STORE DATES IN EAT
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

db = SQLAlchemy()

class BaseModel(db.Model):
    __abstract__ = True
    
    # Use UUID for ID - Use a callable function
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # IMPORTANT: created_at and updated_at now store EAT timestamps directly
    # No UTC conversion - these are EAT naive datetimes
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def save(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
    
    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save()
    
    def to_dict(self):
        """Convert model to dictionary"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
    
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.id}>"


# Import all models after db is defined to avoid circular imports
from .task_models import (
    Task, 
    TaskComment, 
    Attachment as TaskAttachment,  # Alias Attachment to TaskAttachment for compatibility
    TaskHistory, 
    TaskActivity,
    TaskDependency,
    TaskWorkflow,
    Escalation,
    WorkflowTemplate, 
    WorkflowStep,
    TaskTemplate,
    KnowledgeBase
)

from .user_models import (
    User,              # Main user model
    UserPreference,    # User preferences
    Department,        # Department model
    Notification,      # Notifications
    ActivityLog,       # Activity logs
    AuditLog,          # Audit logs
    UserSession,       # User sessions
    Skill,             # Skills
    UserSkill,         # User-Skill association
    PasswordResetToken # Password reset tokens
)