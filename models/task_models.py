# models/task_models.py - COMPLETE FIXED VERSION
from .database import db, BaseModel
from datetime import datetime, timedelta
import json
import uuid

class Task(BaseModel):
    __tablename__ = 'tasks'
    
    # Basic Information
    task_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    
    # Classification
    category = db.Column(db.String(100), nullable=False, index=True)  # FIXED: Added nullable=False
    subcategory = db.Column(db.String(100))
    priority = db.Column(db.String(20), nullable=False, index=True)
    complexity = db.Column(db.String(20), default='Medium')
    
    # Assignment
    assigned_to = db.Column(db.String(36), db.ForeignKey('users.id'), index=True)
    assigned_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    assigned_at = db.Column(db.DateTime)
    secondary_assignees = db.Column(db.JSON, default=list)
    
    # Status Tracking
    status = db.Column(db.String(50), default='New', index=True)
    progress = db.Column(db.Integer, default=0)
    
    # Time Management
    due_date = db.Column(db.DateTime, index=True)
    estimated_hours = db.Column(db.Float, default=1.0)
    actual_hours = db.Column(db.Float, default=0.0)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    
    # SLA Tracking
    sla_level = db.Column(db.String(20))
    sla_due_date = db.Column(db.DateTime, index=True)
    sla_status = db.Column(db.String(20), default='Within SLA')
    
    # Department Information
    department = db.Column(db.String(100), nullable=False, index=True)
    location = db.Column(db.String(200))
    
    # Workflow
    workflow_type = db.Column(db.String(100))
    current_step = db.Column(db.Integer, default=1)
    total_steps = db.Column(db.Integer, default=1)
    
    # Checklist
    checklist = db.Column(db.JSON, default=list)
    
    # Attachments
    attachments = db.Column(db.JSON, default=list)
    
    # Resolution
    resolution_summary = db.Column(db.Text)
    root_cause = db.Column(db.Text)
    preventive_action = db.Column(db.Text)
    resolution_code = db.Column(db.String(100))
    
    # Feedback
    requester_feedback = db.Column(db.Text)
    rating = db.Column(db.Integer)
    feedback_submitted_at = db.Column(db.DateTime)
    
    # Audit
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    closed_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    closed_at = db.Column(db.DateTime)
    
    # Additional fields
    tags = db.Column(db.JSON, default=list)
    is_public = db.Column(db.Boolean, default=True)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    assigned_to_user = db.relationship('User', 
                                      foreign_keys=[assigned_to], 
                                      backref=db.backref('tasks_assigned_to_me', lazy='dynamic'),
                                      lazy=True)
    
    assigned_by_user = db.relationship('User', 
                                      foreign_keys=[assigned_by], 
                                      backref=db.backref('tasks_assigned_by_me', lazy='dynamic'),
                                      lazy=True)
    
    created_by_user = db.relationship('User', 
                                     foreign_keys=[created_by], 
                                     backref=db.backref('tasks_created_by_me', lazy='dynamic'),
                                     lazy=True)
    
    closed_by_user = db.relationship('User', 
                                    foreign_keys=[closed_by], 
                                    backref=db.backref('tasks_closed_by_me', lazy='dynamic'),
                                    lazy=True)
    
    def __init__(self, **kwargs):
        # Call parent constructor first
        super().__init__(**kwargs)
        
        # Generate task_id if not provided
        if not self.task_id:
            self.generate_task_id()
        
        # Calculate SLA if priority is provided
        if 'priority' in kwargs:
            self.calculate_sla()
    
    def generate_task_id(self):
        """Generate sequential task ID"""
        from datetime import datetime
        year = datetime.now().year
        last_task = Task.query.order_by(Task.created_at.desc()).first()
        if last_task and last_task.task_id:
            try:
                parts = last_task.task_id.split('-')
                if len(parts) >= 3:
                    last_number = int(parts[-1])
                    new_number = last_number + 1
                else:
                    new_number = 1
            except (ValueError, IndexError):
                new_number = 1
        else:
            new_number = 1
        
        self.task_id = f"IT-{year}-{new_number:04d}"
    
    def calculate_sla(self):
        """Calculate SLA due date based on priority"""
        from datetime import datetime, timedelta
        
        sla_hours = {
            'Critical': 4,
            'High': 8,
            'Medium': 24,
            'Low': 72
        }
        
        if self.priority in sla_hours:
            hours = sla_hours[self.priority]
            self.sla_level = self.priority
            self.sla_due_date = datetime.utcnow() + timedelta(hours=hours)
    
    @property
    def assigned_to_display(self):
        """Get the display name of the assigned user"""
        if self.assigned_to_user:
            return self.assigned_to_user.full_name
        return "Unassigned"
    
    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'subcategory': self.subcategory,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'assigned_to': self.assigned_to,
            'assigned_to_display': self.assigned_to_display,
            'department': self.department,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'sla_due_date': self.sla_due_date.isoformat() if self.sla_due_date else None,
            'sla_status': self.sla_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
            'estimated_hours': self.estimated_hours,
            'actual_hours': self.actual_hours,
            'tags': self.tags,
            'is_public': self.is_public,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def __repr__(self):
        return f'<Task {self.task_id}: {self.title}>'


class TaskHistory(BaseModel):
    """Track complete history of task changes"""
    __tablename__ = 'task_history'
    
    task_id = db.Column(db.String(36), db.ForeignKey('tasks.id'), nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    action = db.Column(db.String(100), nullable=False)
    field_changed = db.Column(db.String(100))
    old_value = db.Column(db.Text)
    new_value = db.Column(db.Text)
    description = db.Column(db.Text)
    
    # Relationships
    task = db.relationship('Task', 
                          backref=db.backref('history_entries', 
                                           lazy='dynamic',
                                           cascade='all, delete-orphan',
                                           order_by='TaskHistory.created_at.desc()'))
    
    user = db.relationship('User', 
                          backref=db.backref('task_history_entries', lazy='dynamic'),
                          lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'action': self.action,
            'field_changed': self.field_changed,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'description': self.description,
            'timestamp': self.created_at.isoformat() if self.created_at else None,
            'user_name': self.user.full_name if self.user else 'Unknown',
            'user_role': self.user.role if self.user else 'Unknown'
        }
    
    def __repr__(self):
        return f'<TaskHistory {self.id}: {self.action} for Task {self.task_id}>'


class TaskActivity(BaseModel):
    __tablename__ = 'task_activities'
    
    task_id = db.Column(db.String(36), db.ForeignKey('tasks.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    activity_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    old_value = db.Column(db.String(200))
    new_value = db.Column(db.String(200))
    
    time_spent = db.Column(db.Float, default=0.0)
    evidence_files = db.Column(db.JSON, default=list)
    
    # Relationships
    task = db.relationship('Task', 
                          backref=db.backref('activities', 
                                           lazy='dynamic',
                                           cascade='all, delete-orphan'))
    
    user = db.relationship('User', 
                          backref=db.backref('activities', lazy='dynamic'),
                          lazy=True)
    
    def __repr__(self):
        return f'<TaskActivity {self.id}: {self.activity_type} by User {self.user_id}>'


class TaskComment(BaseModel):
    __tablename__ = 'task_comments'
    
    task_id = db.Column(db.String(36), db.ForeignKey('tasks.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    content = db.Column(db.Text, nullable=False)
    is_internal = db.Column(db.Boolean, default=False)
    
    # Relationships
    task = db.relationship('Task', 
                          backref=db.backref('comments', 
                                           lazy='dynamic',
                                           cascade='all, delete-orphan'))
    
    user = db.relationship('User', 
                          backref=db.backref('comments', lazy='dynamic'),
                          lazy=True)
    
    def __repr__(self):
        return f'<TaskComment {self.id} by User {self.user_id}>'


class Escalation(BaseModel):
    __tablename__ = 'escalations'
    
    task_id = db.Column(db.String(36), db.ForeignKey('tasks.id'), nullable=False)
    escalated_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    escalated_to = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    reason = db.Column(db.Text, nullable=False)
    level = db.Column(db.Integer, default=1)
    status = db.Column(db.String(20), default='Pending')
    priority = db.Column(db.String(20), default='High')
    
    # Timestamps
    responded_at = db.Column(db.DateTime)
    resolved_at = db.Column(db.DateTime)
    
    # Relationships
    escalated_by_user = db.relationship('User', 
                                       foreign_keys=[escalated_by], 
                                       backref=db.backref('escalations_made', lazy='dynamic'),
                                       lazy=True)
    
    escalated_to_user = db.relationship('User', 
                                       foreign_keys=[escalated_to], 
                                       backref=db.backref('escalations_received', lazy='dynamic'),
                                       lazy=True)
    
    def __repr__(self):
        return f'<Escalation {self.id}: Level {self.level} for Task {self.task_id}>'


class Report(BaseModel):
    __tablename__ = 'reports'
    
    report_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)
    
    parameters = db.Column(db.JSON, nullable=False)
    generated_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    file_path = db.Column(db.String(500))
    format = db.Column(db.String(10))
    
    data = db.Column(db.JSON)
    
    generated_by_user = db.relationship('User', 
                                       backref=db.backref('reports', lazy='dynamic'),
                                       lazy=True)
    
    def __repr__(self):
        return f'<Report {self.report_id}: {self.name}>'


class WorkflowTemplate(BaseModel):
    __tablename__ = 'workflow_templates'
    
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    steps = db.Column(db.JSON, nullable=False)
    checklist_items = db.Column(db.JSON, default=list)
    
    estimated_time = db.Column(db.Float, default=1.0)
    required_skills = db.Column(db.JSON, default=list)
    
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    creator = db.relationship('User', 
                             backref=db.backref('workflow_templates', lazy='dynamic'),
                             lazy=True)
    
    def __repr__(self):
        return f'<WorkflowTemplate {self.name}>'


class WorkflowStep(BaseModel):
    __tablename__ = 'workflow_steps'
    
    template_id = db.Column(db.String(36), db.ForeignKey('workflow_templates.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    step_order = db.Column(db.Integer, nullable=False)
    assignee_type = db.Column(db.String(50), default='task_creator')
    estimated_duration = db.Column(db.Float)
    
    # Relationship - FIXED: Changed backref name
    template = db.relationship('WorkflowTemplate', 
                              backref=db.backref('template_steps', 
                                               lazy='dynamic',
                                               cascade='all, delete-orphan'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'step_order': self.step_order,
            'assignee_type': self.assignee_type,
            'estimated_duration': self.estimated_duration
        }
    
    def __repr__(self):
        return f'<WorkflowStep {self.name} (Order: {self.step_order})>'


class Attachment(BaseModel):
    __tablename__ = 'attachments'
    
    task_id = db.Column(db.String(36), db.ForeignKey('tasks.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    uploaded_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    description = db.Column(db.String(500))
    is_deleted = db.Column(db.Boolean, default=False)
    
    task = db.relationship('Task', 
                          backref=db.backref('file_attachments', 
                                           lazy='dynamic',
                                           cascade='all, delete-orphan'))
    
    uploaded_by_user = db.relationship('User', 
                                      backref=db.backref('uploaded_attachments', lazy='dynamic'),
                                      lazy=True)
    
    def __repr__(self):
        return f'<Attachment {self.filename} for Task {self.task_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'uploaded_by': self.uploaded_by,
            'uploaded_at': self.created_at.isoformat() if self.created_at else None,
            'description': self.description
        }


class TaskTemplate(BaseModel):
    __tablename__ = 'task_templates'
    
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(100), nullable=False)
    priority = db.Column(db.String(20), default='Medium')
    estimated_hours = db.Column(db.Float, default=1.0)
    checklist = db.Column(db.JSON, default=list)
    required_skills = db.Column(db.JSON, default=list)
    department = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    creator = db.relationship('User', 
                             backref=db.backref('task_templates', lazy='dynamic'),
                             lazy=True)
    
    def __repr__(self):
        return f'<TaskTemplate {self.name}>'


class KnowledgeBase(BaseModel):
    __tablename__ = 'knowledge_base'
    
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    tags = db.Column(db.JSON, default=list)
    views = db.Column(db.Integer, default=0)
    helpful_count = db.Column(db.Integer, default=0)
    not_helpful_count = db.Column(db.Integer, default=0)
    is_published = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    creator = db.relationship('User', 
                             backref=db.backref('knowledge_base_articles', lazy='dynamic'),
                             lazy=True)
    
    def __repr__(self):
        return f'<KnowledgeBase {self.title}>'


# Helper functions
def initialize_default_workflows():
    """Create default workflow templates if they don't exist"""
    from models.user_models import User
    from models.database import db
    
    default_workflows = [
        {
            'name': 'Standard IT Support',
            'category': 'IT Support',
            'description': 'Standard workflow for IT support requests',
            'steps': [
                {'name': 'Initial Assessment', 'description': 'Assess the issue', 'estimated_time': 0.5},
                {'name': 'Troubleshooting', 'description': 'Perform troubleshooting', 'estimated_time': 1.0},
                {'name': 'Resolution', 'description': 'Implement solution', 'estimated_time': 0.5},
                {'name': 'Verification', 'description': 'Verify resolution with user', 'estimated_time': 0.25}
            ],
            'checklist_items': [
                'Issue properly categorized',
                'User information collected',
                'Impact assessment completed',
                'Resolution documented'
            ],
            'estimated_time': 2.25,
            'required_skills': ['Troubleshooting', 'Communication']
        }
    ]
    
    admin = User.query.filter_by(email='admin@rileyfalcon.com').first()
    
    if admin:
        for workflow_data in default_workflows:
            existing = WorkflowTemplate.query.filter_by(name=workflow_data['name']).first()
            if not existing:
                workflow = WorkflowTemplate(
                    name=workflow_data['name'],
                    category=workflow_data['category'],
                    description=workflow_data['description'],
                    steps=workflow_data['steps'],
                    checklist_items=workflow_data['checklist_items'],
                    estimated_time=workflow_data['estimated_time'],
                    required_skills=workflow_data['required_skills'],
                    created_by=admin.id
                )
                db.session.add(workflow)
        
        db.session.commit()
        print("Default workflow templates created")
    
    return True


def log_task_history(task_id, user_id, action, field_changed=None, old_value=None, new_value=None, description=None):
    """Log a task change to history"""
    from models.database import db
    
    # Handle None values properly
    old_val = str(old_value) if old_value is not None else None
    new_val = str(new_value) if new_value is not None else None
    
    history = TaskHistory(
        task_id=task_id,
        user_id=user_id,
        action=action,
        field_changed=field_changed,
        old_value=old_val,
        new_value=new_val,
        description=description
    )
    
    db.session.add(history)
    return history


def get_task_workflow(task_id):
    """Get complete workflow timeline for a task"""
    task_history = TaskHistory.query.filter_by(task_id=task_id)\
        .order_by(TaskHistory.created_at.asc())\
        .all()
    
    # Enrich with user information
    workflow_data = []
    for history in task_history:
        item = history.to_dict()
        workflow_data.append(item)
    
    return workflow_data


def log_task_creation(task, user_id):
    """Log task creation to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Task Created',
        description=f'Task "{task.title}" created with ID {task.task_id}',
        new_value=task.status
    )


def log_task_assignment(task, user_id, old_assignee=None, new_assignee=None):
    """Log task assignment to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Task Assigned',
        field_changed='assigned_to',
        old_value=old_assignee,
        new_value=new_assignee,
        description=f'Task assigned from {old_assignee or "Unassigned"} to {new_assignee or "Unassigned"}'
    )


def log_status_change(task, user_id, old_status, new_status):
    """Log task status change to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Status Changed',
        field_changed='status',
        old_value=old_status,
        new_value=new_status,
        description=f'Status changed from {old_status} to {new_status}'
    )


def log_task_closure(task, user_id, resolution_summary):
    """Log task closure to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Task Closed',
        field_changed='status',
        old_value=task.status,
        new_value='Resolved',
        description=f'Task marked as resolved. Resolution: {resolution_summary[:200]}...'
    )


def log_priority_change(task, user_id, old_priority, new_priority):
    """Log task priority change to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Priority Changed',
        field_changed='priority',
        old_value=old_priority,
        new_value=new_priority,
        description=f'Priority changed from {old_priority} to {new_priority}'
    )


def log_due_date_change(task, user_id, old_due_date, new_due_date):
    """Log task due date change to history"""
    old_date_str = old_due_date.strftime('%Y-%m-%d') if old_due_date else 'Not set'
    new_date_str = new_due_date.strftime('%Y-%m-%d') if new_due_date else 'Not set'
    
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Due Date Updated',
        field_changed='due_date',
        old_value=old_date_str,
        new_value=new_date_str,
        description=f'Due date updated from {old_date_str} to {new_date_str}'
    )


def log_task_comment(task_id, user_id, comment):
    """Log task comment to history"""
    return log_task_history(
        task_id=task_id,
        user_id=user_id,
        action='Comment Added',
        description=f'Comment added: {comment[:100]}...'
    )


def log_task_update(task, user_id, field_changed, old_value, new_value):
    """Log general task update to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Task Updated',
        field_changed=field_changed,
        old_value=str(old_value) if old_value is not None else None,
        new_value=str(new_value) if new_value is not None else None,
        description=f'{field_changed.replace("_", " ").title()} updated from "{old_value}" to "{new_value}"'
    )


def log_task_reopen(task, user_id, reason):
    """Log task reopening to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Task Reopened',
        field_changed='status',
        old_value='Resolved',
        new_value='In Progress',
        description=f'Task reopened. Reason: {reason}'
    )


def log_task_escalation(task_id, user_id, escalated_to, reason):
    """Log task escalation to history"""
    from models.user_models import User
    escalated_to_user = User.query.get(escalated_to)
    escalated_to_name = escalated_to_user.full_name if escalated_to_user else 'Unknown'
    
    return log_task_history(
        task_id=task_id,
        user_id=user_id,
        action='Task Escalated',
        description=f'Task escalated to {escalated_to_name}. Reason: {reason}'
    )


def log_checklist_update(task, user_id, checklist_item, status):
    """Log checklist item update to history"""
    return log_task_history(
        task_id=task.id,
        user_id=user_id,
        action='Checklist Updated',
        description=f'Checklist item "{checklist_item}" marked as {status}'
    )