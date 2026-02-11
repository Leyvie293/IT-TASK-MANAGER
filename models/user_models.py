# models/user_models.py - COMPLETE FIXED VERSION
from .database import db, BaseModel
from flask_login import UserMixin
from flask_bcrypt import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

class User(UserMixin, BaseModel):
    __tablename__ = 'users'
    
    # Personal Information
    employee_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(20))
    
    # Authentication
    password_hash = db.Column(db.String(128), nullable=False)
    
    # Role and Department
    role = db.Column(db.String(50), nullable=False, default='Technician')  # Admin, Supervisor, Technician, Requester, Manager
    department = db.Column(db.String(100), nullable=False, default='IT')
    job_title = db.Column(db.String(100))
    
    # Work Information
    skills = db.Column(db.JSON, default=list)
    max_tasks = db.Column(db.Integer, default=5)
    is_available = db.Column(db.Boolean, default=True, index=True)
    
    # IMPORTANT: is_active as a database column
    is_active = db.Column(db.Boolean, default=True, nullable=False, index=True)
    
    # Audit
    last_login = db.Column(db.DateTime)
    login_count = db.Column(db.Integer, default=0)
    last_password_change = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, **kwargs):
        # Handle password separately
        password = kwargs.pop('password', None)
        
        # Call parent constructor FIRST to get UUID ID
        super().__init__(**kwargs)
        
        # Now handle password if provided
        if password:
            self.password = password  # This will use the password setter
        
        # Ensure employee_id is generated if not provided
        if not self.employee_id:
            self.employee_id = self._generate_employee_id()
    
    def _generate_employee_id(self):
        """Generate employee ID based on role"""
        role_prefix = {
            'Admin': 'RFSS-ADMIN',
            'Supervisor': 'RFSS-SUP',
            'Technician': 'RFSS-TECH',
            'Manager': 'RFSS-MGR',
            'Requester': 'RFSS-REQ'
        }.get(self.role, 'RFSS-EMP')
        
        # Find last employee ID with this prefix
        last_user = User.query.filter(User.employee_id.like(f'{role_prefix}-%')).order_by(User.created_at.desc()).first()
        if last_user and last_user.employee_id:
            try:
                last_num = int(last_user.employee_id.split('-')[-1])
                new_num = last_num + 1
            except (ValueError, IndexError):
                new_num = 1
        else:
            new_num = 1
        
        return f"{role_prefix}-{new_num:03d}"
    
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password).decode('utf-8')
        self.last_password_change = datetime.utcnow()
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def initials(self):
        return f"{self.first_name[0]}{self.last_name[0]}" if self.first_name and self.last_name else "??"
    
    @property
    def task_count(self):
        """Get number of tasks assigned to this user"""
        from models.task_models import Task
        return Task.query.filter_by(assigned_to=self.id).count()
    
    @property
    def open_task_count(self):
        """Get number of open tasks assigned to this user"""
        from models.task_models import Task
        return Task.query.filter_by(assigned_to=self.id)\
            .filter(Task.status.in_(['New', 'Assigned', 'In Progress']))\
            .count()
    
    @property
    def overdue_task_count(self):
        """Get number of overdue tasks assigned to this user"""
        from models.task_models import Task
        from datetime import datetime
        return Task.query.filter(
            Task.assigned_to == self.id,
            Task.due_date < datetime.utcnow(),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
    
    def is_admin(self):
        return self.role == 'Admin'
    
    def is_supervisor(self):
        return self.role in ['Admin', 'Supervisor']
    
    def is_technician(self):
        return self.role in ['Admin', 'Supervisor', 'Technician']
    
    def can_assign_tasks(self):
        return self.role in ['Admin', 'Supervisor', 'Manager']
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'full_name': self.full_name,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'role': self.role,
            'department': self.department,
            'job_title': self.job_title,
            'is_available': self.is_available,
            'is_active': self.is_active,
            'skills': self.skills,
            'max_tasks': self.max_tasks,
            'task_count': self.task_count,
            'open_task_count': self.open_task_count,
            'overdue_task_count': self.overdue_task_count,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_count': self.login_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_minimal_dict(self):
        """Return minimal user info for dropdowns and lists"""
        return {
            'id': self.id,
            'full_name': self.full_name,
            'email': self.email,
            'role': self.role,
            'department': self.department,
            'is_available': self.is_available,
            'employee_id': self.employee_id
        }
    
    def __repr__(self):
        return f'<User {self.employee_id}: {self.full_name} ({self.role})>'


class UserPreference(BaseModel):
    __tablename__ = 'user_preferences'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, unique=True)
    
    # Notification preferences
    email_notifications = db.Column(db.Boolean, default=True)
    task_assigned_email = db.Column(db.Boolean, default=True)
    task_updated_email = db.Column(db.Boolean, default=True)
    task_completed_email = db.Column(db.Boolean, default=True)
    overdue_task_email = db.Column(db.Boolean, default=True)
    
    # SMS notifications
    sms_notifications = db.Column(db.Boolean, default=False)
    urgent_task_sms = db.Column(db.Boolean, default=True)
    
    # UI preferences
    theme = db.Column(db.String(20), default='dark')  # light, dark, auto
    language = db.Column(db.String(10), default='en')
    items_per_page = db.Column(db.Integer, default=20)
    default_view = db.Column(db.String(50), default='list')  # list, card, calendar
    compact_mode = db.Column(db.Boolean, default=False)
    
    # Email frequency
    daily_summary = db.Column(db.Boolean, default=True)
    weekly_report = db.Column(db.Boolean, default=True)
    send_time = db.Column(db.String(5), default='08:00')  # HH:MM format
    
    user = db.relationship('User', backref=db.backref('preference', uselist=False, cascade="all, delete-orphan"), lazy=True)
    
    def __repr__(self):
        return f'<UserPreference for User {self.user_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'email_notifications': self.email_notifications,
            'task_assigned_email': self.task_assigned_email,
            'task_updated_email': self.task_updated_email,
            'task_completed_email': self.task_completed_email,
            'overdue_task_email': self.overdue_task_email,
            'sms_notifications': self.sms_notifications,
            'urgent_task_sms': self.urgent_task_sms,
            'theme': self.theme,
            'language': self.language,
            'items_per_page': self.items_per_page,
            'default_view': self.default_view,
            'compact_mode': self.compact_mode,
            'daily_summary': self.daily_summary,
            'weekly_report': self.weekly_report,
            'send_time': self.send_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Department(BaseModel):
    __tablename__ = 'departments'
    
    name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    code = db.Column(db.String(10), unique=True, nullable=False, index=True)
    description = db.Column(db.Text)
    manager_id = db.Column(db.String(36), db.ForeignKey('users.id'))
    contact_email = db.Column(db.String(120))
    contact_phone = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    
    manager = db.relationship('User', backref='managed_departments', lazy=True)
    
    @property
    def member_count(self):
        return User.query.filter_by(department=self.name, is_active=True).count()
    
    @property
    def active_member_count(self):
        return User.query.filter_by(department=self.name, is_active=True, is_available=True).count()
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'description': self.description,
            'manager_id': self.manager_id,
            'manager_name': self.manager.full_name if self.manager else None,
            'contact_email': self.contact_email,
            'contact_phone': self.contact_phone,
            'is_active': self.is_active,
            'member_count': self.member_count,
            'active_member_count': self.active_member_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Department {self.code}: {self.name}>'


class Notification(BaseModel):
    __tablename__ = 'notifications'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column(db.String(50), nullable=False)  # task_assigned, task_updated, system, alert
    related_type = db.Column(db.String(50))  # task, user, system
    related_id = db.Column(db.String(50))
    priority = db.Column(db.String(20), default='normal')  # low, normal, high, urgent
    is_read = db.Column(db.Boolean, default=False, index=True)
    read_at = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref='notifications', lazy=True)
    
    def mark_as_read(self):
        self.is_read = True
        self.read_at = datetime.utcnow()
        from .database import db
        db.session.commit()
    
    def is_expired(self):
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'notification_type': self.notification_type,
            'related_type': self.related_type,
            'related_id': self.related_id,
            'priority': self.priority,
            'is_read': self.is_read,
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_expired': self.is_expired()
        }
    
    def __repr__(self):
        return f'<Notification for User {self.user_id}: {self.title}>'


class ActivityLog(BaseModel):
    __tablename__ = 'activity_logs'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True, index=True)
    activity_type = db.Column(db.String(100), nullable=False, index=True)  # e.g., 'User Login', 'Task Created'
    description = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), nullable=True, index=True)  # 'success', 'failed', 'warning'
    old_value = db.Column(db.Text, nullable=True)
    new_value = db.Column(db.Text, nullable=True)
    additional_data = db.Column(db.JSON, default=dict)
    
    # Relationships
    user = db.relationship('User', backref='activity_logs', lazy=True)
    
    def __repr__(self):
        return f'<ActivityLog {self.id}: {self.activity_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.full_name if self.user else 'System',
            'user_email': self.user.email if self.user else 'System',
            'activity_type': self.activity_type,
            'description': self.description,
            'ip_address': self.ip_address,
            'status': self.status,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'additional_data': self.additional_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AuditLog(BaseModel):
    __tablename__ = 'audit_logs'
    
    table_name = db.Column(db.String(100), nullable=False, index=True)
    record_id = db.Column(db.String(50), nullable=False, index=True)
    action = db.Column(db.String(50), nullable=False)  # CREATE, UPDATE, DELETE
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    old_values = db.Column(db.JSON, nullable=True)
    new_values = db.Column(db.JSON, nullable=True)
    changes = db.Column(db.JSON, nullable=True)
    
    user = db.relationship('User', backref='audit_logs', lazy=True)
    
    def __repr__(self):
        return f'<AuditLog {self.table_name}.{self.record_id}: {self.action}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'table_name': self.table_name,
            'record_id': self.record_id,
            'action': self.action,
            'user_id': self.user_id,
            'user_name': self.user.full_name if self.user else 'System',
            'ip_address': self.ip_address,
            'old_values': self.old_values,
            'new_values': self.new_values,
            'changes': self.changes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class UserSession(BaseModel):
    __tablename__ = 'user_sessions'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    session_id = db.Column(db.String(255), nullable=False, unique=True, index=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True, index=True)
    
    user = db.relationship('User', backref='sessions', lazy=True)
    
    def update_activity(self):
        self.last_activity = datetime.utcnow()
        from .database import db
        db.session.commit()
    
    def is_expired(self):
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def end_session(self):
        self.is_active = False
        from .database import db
        db.session.commit()
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'login_time': self.login_time.isoformat() if self.login_time else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'is_expired': self.is_expired(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<UserSession {self.session_id[:8]}... for User {self.user_id}>'


class Skill(BaseModel):
    __tablename__ = 'skills'
    
    name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    category = db.Column(db.String(50), nullable=True, index=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True, index=True)
    
    # Many-to-many relationship with users
    users = db.relationship('User', secondary='user_skills', backref='skill_objects')
    
    def __repr__(self):
        return f'<Skill {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'is_active': self.is_active,
            'user_count': len(self.users),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# Association table for many-to-many relationship between users and skills
class UserSkill(BaseModel):
    __tablename__ = 'user_skills'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), primary_key=True)
    skill_id = db.Column(db.String(36), db.ForeignKey('skills.id'), primary_key=True)
    proficiency = db.Column(db.String(20), default='intermediate')  # beginner, intermediate, advanced, expert
    years_experience = db.Column(db.Integer, default=1)
    certified = db.Column(db.Boolean, default=False)
    certification_date = db.Column(db.DateTime, nullable=True)
    
    user = db.relationship('User', backref=db.backref('skill_associations', lazy='dynamic'))
    skill = db.relationship('Skill', backref=db.backref('user_associations', lazy='dynamic'))
    
    def __repr__(self):
        return f'<UserSkill User:{self.user_id} Skill:{self.skill_id}>'
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'skill_id': self.skill_id,
            'proficiency': self.proficiency,
            'years_experience': self.years_experience,
            'certified': self.certified,
            'certification_date': self.certification_date.isoformat() if self.certification_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PasswordResetToken(BaseModel):
    __tablename__ = 'password_reset_tokens'
    
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    used_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    
    user = db.relationship('User', backref='password_reset_tokens', lazy=True)
    
    def is_valid(self):
        return not self.used and datetime.utcnow() < self.expires_at
    
    def mark_as_used(self):
        self.used = True
        self.used_at = datetime.utcnow()
        from .database import db
        db.session.commit()
    
    def __repr__(self):
        return f'<PasswordResetToken for User {self.user_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'token': self.token,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'used': self.used,
            'used_at': self.used_at.isoformat() if self.used_at else None,
            'ip_address': self.ip_address,
            'is_valid': self.is_valid(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_default_admin():
    """Create default admin user if none exists"""
    from .database import db
    
    admin = User.query.filter_by(email='admin@rileyfalcon.com').first()
    if not admin:
        # Create admin user first
        admin = User(
            first_name='System',
            last_name='Administrator',
            email='admin@rileyfalcon.com',
            phone='+1234567890',
            password='Admin@123',
            role='Admin',
            department='IT',
            job_title='System Administrator',
            skills=['System Administration', 'IT Management', 'Security'],
            max_tasks=10,
            is_active=True,
            is_available=True,
            employee_id='RFSS-ADMIN-001'  # Explicit ID to avoid generation conflicts
        )
        db.session.add(admin)
        db.session.flush()  # Get the ID without committing
        
        # Create preferences with the user ID
        preferences = UserPreference(
            user_id=admin.id,  # Now admin.id should have a value
            email_notifications=True,
            task_assigned_email=True,
            task_updated_email=True,
            task_completed_email=True,
            overdue_task_email=True,
            sms_notifications=False,
            urgent_task_sms=True,
            theme='dark',
            language='en',
            items_per_page=20,
            default_view='list',
            compact_mode=False,
            daily_summary=True,
            weekly_report=True,
            send_time='08:00'
        )
        db.session.add(preferences)
        db.session.commit()
        print("✓ Default admin user created")
    
    return admin


def create_user_with_preferences(user_data):
    """Helper function to create a user with default preferences"""
    from .database import db
    
    try:
        # Create the user
        user = User(**user_data)
        db.session.add(user)
        db.session.flush()  # Generate the UUID without committing
        
        # Create default preferences
        preferences = UserPreference(
            user_id=user.id,
            email_notifications=True,
            task_assigned_email=True,
            task_updated_email=True,
            task_completed_email=True,
            overdue_task_email=True,
            sms_notifications=False,
            urgent_task_sms=True,
            theme='dark',
            language='en',
            items_per_page=20,
            default_view='list',
            compact_mode=False,
            daily_summary=True,
            weekly_report=True,
            send_time='08:00'
        )
        db.session.add(preferences)
        db.session.commit()
        return user
    except Exception as e:
        db.session.rollback()
        print(f"Error creating user: {e}")
        return None


def create_sample_users():
    """Create sample users for testing"""
    from .database import db
    
    sample_users = [
        {
            'first_name': 'IT',
            'last_name': 'Supervisor',
            'email': 'supervisor@rileyfalcon.com',
            'password': 'Super@123',
            'role': 'Supervisor',
            'department': 'IT',
            'job_title': 'IT Supervisor',
            'skills': ['Supervision', 'IT Management', 'Troubleshooting']
        },
        {
            'first_name': 'John',
            'last_name': 'Technician',
            'email': 'tech@rileyfalcon.com',
            'password': 'Tech@123',
            'role': 'Technician',
            'department': 'IT',
            'job_title': 'IT Technician',
            'skills': ['Hardware', 'Software', 'Networking']
        },
        {
            'first_name': 'Sarah',
            'last_name': 'Manager',
            'email': 'manager@rileyfalcon.com',
            'password': 'Manager@123',
            'role': 'Manager',
            'department': 'IT',
            'job_title': 'IT Manager',
            'skills': ['Management', 'Budgeting', 'Planning']
        }
    ]
    
    created_count = 0
    for user_data in sample_users:
        existing = User.query.filter_by(email=user_data['email']).first()
        if not existing:
            user = create_user_with_preferences(user_data)
            if user:
                created_count += 1
    
    if created_count > 0:
        print(f"✓ Created {created_count} sample users")
    
    return created_count


def initialize_default_departments():
    """Create default departments if they don't exist"""
    from .database import db
    
    default_departments = [
        {'name': 'Information Technology', 'code': 'IT', 'description': 'IT Support and Infrastructure'},
        {'name': 'Human Resources', 'code': 'HR', 'description': 'Human Resources Department'},
        {'name': 'Finance', 'code': 'FIN', 'description': 'Finance and Accounting'},
        {'name': 'Operations', 'code': 'OPS', 'description': 'Operations Management'},
        {'name': 'Security', 'code': 'SEC', 'description': 'Physical Security'},
        {'name': 'Administration', 'code': 'ADMIN', 'description': 'Administrative Support'},
        {'name': 'Marketing', 'code': 'MKT', 'description': 'Marketing and Communications'},
        {'name': 'Sales', 'code': 'SALES', 'description': 'Sales Department'},
        {'name': 'Customer Support', 'code': 'SUPPORT', 'description': 'Customer Support'},
        {'name': 'Research & Development', 'code': 'RND', 'description': 'Research and Development'}
    ]
    
    for dept_data in default_departments:
        existing = Department.query.filter_by(code=dept_data['code']).first()
        if not existing:
            department = Department(
                name=dept_data['name'],
                code=dept_data['code'],
                description=dept_data['description']
            )
            db.session.add(department)
    
    db.session.commit()
    print("✓ Default departments created")


def initialize_default_skills():
    """Create default skills if they don't exist"""
    from .database import db
    
    default_skills = [
        {'name': 'Python', 'category': 'Programming'},
        {'name': 'JavaScript', 'category': 'Programming'},
        {'name': 'SQL', 'category': 'Database'},
        {'name': 'Network Administration', 'category': 'IT'},
        {'name': 'System Administration', 'category': 'IT'},
        {'name': 'Help Desk Support', 'category': 'IT'},
        {'name': 'Project Management', 'category': 'Management'},
        {'name': 'Customer Service', 'category': 'Soft Skills'},
        {'name': 'Troubleshooting', 'category': 'IT'},
        {'name': 'Documentation', 'category': 'Soft Skills'},
        {'name': 'Flask', 'category': 'Web Development'},
        {'name': 'HTML/CSS', 'category': 'Web Development'},
        {'name': 'Linux', 'category': 'Operating Systems'},
        {'name': 'Windows Server', 'category': 'Operating Systems'},
        {'name': 'Cloud Computing', 'category': 'IT'},
        {'name': 'Cybersecurity', 'category': 'Security'},
        {'name': 'Data Analysis', 'category': 'Analytics'},
        {'name': 'Communication', 'category': 'Soft Skills'},
        {'name': 'Team Leadership', 'category': 'Management'},
        {'name': 'Time Management', 'category': 'Soft Skills'}
    ]
    
    for skill_data in default_skills:
        existing = Skill.query.filter_by(name=skill_data['name']).first()
        if not existing:
            skill = Skill(
                name=skill_data['name'],
                category=skill_data['category'],
                description=f'Skill in {skill_data["name"]}',
                is_active=True
            )
            db.session.add(skill)
    
    db.session.commit()
    print("✓ Default skills created")


def initialize_all_defaults():
    """Initialize all default data"""
    print("\n" + "="*50)
    print("INITIALIZING DEFAULT DATA")
    print("="*50)
    
    admin = create_default_admin()
    create_sample_users()
    initialize_default_departments()
    initialize_default_skills()
    
    print("✓ All default data initialized")
    return admin


def get_active_users():
    """Get all active users"""
    return User.query.filter_by(is_active=True).order_by(User.first_name, User.last_name).all()


def get_available_users():
    """Get all active and available users"""
    return User.query.filter_by(is_active=True, is_available=True).order_by(User.first_name, User.last_name).all()


def get_users_by_department(department_name):
    """Get all users in a specific department"""
    return User.query.filter_by(department=department_name, is_active=True)\
        .order_by(User.first_name, User.last_name).all()


def get_users_by_role(role):
    """Get all users with a specific role"""
    return User.query.filter_by(role=role, is_active=True)\
        .order_by(User.first_name, User.last_name).all()


def search_users(query, limit=20):
    """Search users by name, email, or employee ID"""
    return User.query.filter(
        (User.first_name.ilike(f'%{query}%')) |
        (User.last_name.ilike(f'%{query}%')) |
        (User.email.ilike(f'%{query}%')) |
        (User.employee_id.ilike(f'%{query}%'))
    ).filter_by(is_active=True).limit(limit).all()


def get_user_by_email(email):
    """Get user by email"""
    return User.query.filter_by(email=email, is_active=True).first()


def get_user_by_employee_id(employee_id):
    """Get user by employee ID"""
    return User.query.filter_by(employee_id=employee_id, is_active=True).first()


def create_user_activity_log(user_id, activity_type, description=None, ip_address=None, user_agent=None, status='success', old_value=None, new_value=None, additional_data=None):
    """Create an activity log entry"""
    from .database import db
    
    log = ActivityLog(
        user_id=user_id,
        activity_type=activity_type,
        description=description,
        ip_address=ip_address,
        user_agent=user_agent,
        status=status,
        old_value=old_value,
        new_value=new_value,
        additional_data=additional_data or {}
    )
    
    db.session.add(log)
    db.session.commit()
    return log


def log_user_login(user, ip_address=None, user_agent=None):
    """Log user login activity"""
    user.last_login = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    
    create_user_activity_log(
        user_id=user.id,
        activity_type='User Login',
        description=f'User {user.email} logged in',
        ip_address=ip_address,
        user_agent=user_agent,
        status='success'
    )
    
    from .database import db
    db.session.commit()


def log_user_logout(user, ip_address=None, user_agent=None):
    """Log user logout activity"""
    create_user_activity_log(
        user_id=user.id,
        activity_type='User Logout',
        description=f'User {user.email} logged out',
        ip_address=ip_address,
        user_agent=user_agent,
        status='success'
    )


def get_user_stats(user_id):
    """Get comprehensive statistics for a user"""
    from models.task_models import Task
    from datetime import datetime, timedelta
    
    user = User.query.get(user_id)
    if not user:
        return None
    
    # Calculate today's date
    today = datetime.utcnow().date()
    
    # Get tasks completed today
    tasks_completed_today = Task.query.filter(
        Task.assigned_to == user_id,
        Task.status == 'Resolved',
        db.func.date(Task.completed_at) == today
    ).count()
    
    # Get tasks created today
    tasks_created_today = Task.query.filter(
        Task.created_by == user_id,
        db.func.date(Task.created_at) == today
    ).count()
    
    # Get average resolution time (in hours)
    completed_tasks = Task.query.filter(
        Task.assigned_to == user_id,
        Task.status == 'Resolved',
        Task.completed_at.isnot(None),
        Task.created_at.isnot(None)
    ).all()
    
    avg_resolution_time = 0
    if completed_tasks:
        total_hours = sum([
            (task.completed_at - task.created_at).total_seconds() / 3600
            for task in completed_tasks if task.completed_at and task.created_at
        ])
        avg_resolution_time = total_hours / len(completed_tasks)
    
    return {
        'user': user.to_minimal_dict(),
        'task_count': user.task_count,
        'open_task_count': user.open_task_count,
        'overdue_task_count': user.overdue_task_count,
        'tasks_completed_today': tasks_completed_today,
        'tasks_created_today': tasks_created_today,
        'avg_resolution_time_hours': round(avg_resolution_time, 2),
        'login_count': user.login_count,
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'is_available': user.is_available,
        'is_active': user.is_active
    }


def ensure_user_preferences(user):
    """Ensure a user has preferences, creating default ones if not"""
    from .database import db
    
    if not user.preference:
        preferences = UserPreference(
            user_id=user.id,
            email_notifications=True,
            task_assigned_email=True,
            task_updated_email=True,
            task_completed_email=True,
            overdue_task_email=True,
            sms_notifications=False,
            urgent_task_sms=True,
            theme='dark',
            language='en',
            items_per_page=20,
            default_view='list',
            compact_mode=False,
            daily_summary=True,
            weekly_report=True,
            send_time='08:00'
        )
        db.session.add(preferences)
        db.session.commit()
    return user.preference


# Debug function to test user creation
def test_user_creation():
    """Test function to verify user creation works"""
    print("Testing user creation...")
    
    # Test creating a simple user
    test_user_data = {
        'first_name': 'Test',
        'last_name': 'User',
        'email': 'test.user@example.com',
        'password': 'Test123!',
        'role': 'Technician',
        'department': 'IT'
    }
    
    user = create_user_with_preferences(test_user_data)
    if user:
        print(f"✓ User created successfully: {user.full_name}")
        print(f"  ID: {user.id}")
        print(f"  Employee ID: {user.employee_id}")
        print(f"  Has preferences: {bool(user.preference)}")
        
        # Clean up test user
        from .database import db
        if user.preference:
            db.session.delete(user.preference)
        db.session.delete(user)
        db.session.commit()
        print("  Test user cleaned up")
    else:
        print("✗ User creation failed")
    
    return user is not None