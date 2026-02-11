# config.py - UPDATED WITH EAT TIMEZONE
import os
from datetime import timedelta
import pytz  # ADDED FOR TIMEZONE SUPPORT

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    APP_NAME = 'IT Task Manager'
    COMPANY_NAME = 'Riley Falcon Security Services'
    APP_VERSION = '1.0.0'
    
    # ====================================================================
    # EAT TIMEZONE CONFIGURATION - ADDED
    # ====================================================================
    DEFAULT_TIMEZONE = 'Africa/Nairobi'  # East Africa Time
    EAT_TIMEZONE = pytz.timezone('Africa/Nairobi')  # EAT timezone object
    
    # Time formatting
    DATE_FORMAT = '%Y-%m-%d'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    TIME_FORMAT = '%H:%M:%S'
    
    # Business hours in EAT (9 AM to 5 PM)
    BUSINESS_HOURS_START = 9  # 9:00 AM EAT
    BUSINESS_HOURS_END = 17   # 5:00 PM EAT
    
    # SLA Settings (in hours) - All times are in EAT context
    DEFAULT_SLA_HOURS = {
        'Critical': 4,    # 4 hours for critical issues
        'High': 8,        # 8 hours for high priority
        'Medium': 24,     # 24 hours for medium priority
        'Low': 48         # 48 hours for low priority
    }
    
    # Work hours per day (for SLA calculations)
    WORK_HOURS_PER_DAY = 8
    # ====================================================================
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Email notifications
    EMAIL_NOTIFICATIONS_ENABLED = True
    NOTIFY_ON_TASK_ASSIGNMENT = True
    NOTIFY_ON_TASK_COMPLETION = True
    NOTIFY_ON_SLA_BREACH = True
    
    # File upload settings
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx'}
    
    # Pagination settings
    ITEMS_PER_PAGE = 20
    MAX_PAGINATION_LINKS = 5
    
    # Report settings
    REPORT_RETENTION_DAYS = 30
    AUTO_EXPORT_REPORTS = False
    
    # Security settings
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_COMPLEXITY = True  # Require uppercase, lowercase, numbers
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    
    # Backup settings
    AUTO_BACKUP_ENABLED = True
    BACKUP_RETENTION_DAYS = 7
    BACKUP_DIR = 'backups'

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///task_manager.db'
    
    # Email settings for development
    MAIL_SERVER = 'localhost'
    MAIL_PORT = 1025
    MAIL_USE_TLS = False
    MAIL_USERNAME = None
    MAIL_PASSWORD = None
    MAIL_DEFAULT_SENDER = 'noreply@rileyfalcon.com'
    
    # Development-specific timezone settings
    FORCE_EAT_TIMEZONE = True  # Always use EAT in development
    
    # Logging
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'logs/app.log'
    
    # Development features
    AUTO_CREATE_ADMIN = True
    SEED_TEST_DATA = True

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///task_manager.db'
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True
    
    # Production email settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@rileyfalcon.com')
    
    # Production timezone settings
    FORCE_EAT_TIMEZONE = True  # Production should also use EAT
    
    # Logging
    LOG_LEVEL = 'WARNING'
    LOG_FILE = '/var/log/it-task-manager/app.log'
    
    # Security
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_COMPLEXITY = True
    MAX_LOGIN_ATTEMPTS = 3
    
    # Backup
    AUTO_BACKUP_ENABLED = True
    BACKUP_RETENTION_DAYS = 30
    
    # Performance
    SQLALCHEMY_POOL_SIZE = 20
    SQLALCHEMY_MAX_OVERFLOW = 30
    SQLALCHEMY_POOL_RECYCLE = 3600


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test-secret-key'
    
    # Testing timezone - use EAT for consistency
    FORCE_EAT_TIMEZONE = True
    
    # Disable features for testing
    EMAIL_NOTIFICATIONS_ENABLED = False
    AUTO_BACKUP_ENABLED = False
    
    # Testing settings
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


# Helper functions for EAT timezone
def get_eat_timezone():
    """Get the EAT timezone object"""
    return pytz.timezone('Africa/Nairobi')


def get_eat_now():
    """Get current datetime in EAT"""
    import datetime
    eat_tz = get_eat_timezone()
    utc_now = datetime.datetime.utcnow()
    return pytz.utc.localize(utc_now).astimezone(eat_tz)


def format_eat_datetime(dt, format_str=None):
    """Format datetime in EAT timezone"""
    if dt is None:
        return ""
    
    eat_tz = get_eat_timezone()
    
    # Ensure dt has timezone info
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = pytz.utc.localize(dt)
    
    # Convert to EAT
    dt_eat = dt.astimezone(eat_tz)
    
    # Format
    if format_str is None:
        format_str = Config.DATETIME_FORMAT
    
    return dt_eat.strftime(format_str)


def format_eat_date(dt, format_str=None):
    """Format date in EAT timezone"""
    if dt is None:
        return ""
    
    eat_tz = get_eat_timezone()
    
    # Ensure dt has timezone info
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = pytz.utc.localize(dt)
    
    # Convert to EAT
    dt_eat = dt.astimezone(eat_tz)
    
    # Format
    if format_str is None:
        format_str = Config.DATE_FORMAT
    
    return dt_eat.strftime(format_str)


def calculate_sla_due_date(priority, created_at=None):
    """
    Calculate SLA due date based on priority and created_at time
    Returns datetime in EAT timezone
    """
    import datetime
    
    eat_tz = get_eat_timezone()
    
    # Use current time if created_at not provided
    if created_at is None:
        created_at = get_eat_now()
    elif created_at.tzinfo is None:
        # Assume UTC if no timezone
        created_at = pytz.utc.localize(created_at)
    
    # Convert to EAT if not already
    if created_at.tzinfo != eat_tz:
        created_at = created_at.astimezone(eat_tz)
    
    # Get SLA hours for priority
    sla_hours = Config.DEFAULT_SLA_HOURS.get(priority, 24)
    
    # Calculate due date (considering business hours)
    due_date = created_at
    
    # Add SLA hours (simplified - doesn't skip weekends/holidays)
    # In a real implementation, you'd want to consider business hours
    due_date = due_date + datetime.timedelta(hours=sla_hours)
    
    return due_date


def is_within_business_hours(dt=None):
    """
    Check if given datetime is within business hours (9 AM - 5 PM EAT)
    """
    if dt is None:
        dt = get_eat_now()
    elif dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = pytz.utc.localize(dt)
    
    # Convert to EAT
    eat_tz = get_eat_timezone()
    dt_eat = dt.astimezone(eat_tz)
    
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if dt_eat.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if within business hours
    hour = dt_eat.hour
    return Config.BUSINESS_HOURS_START <= hour < Config.BUSINESS_HOURS_END


def get_next_business_day(dt=None):
    """
    Get the next business day (Monday-Friday) at 9 AM EAT
    """
    import datetime
    
    if dt is None:
        dt = get_eat_now()
    elif dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = pytz.utc.localize(dt)
    
    # Convert to EAT
    eat_tz = get_eat_timezone()
    dt_eat = dt.astimezone(eat_tz)
    
    # Add one day
    next_day = dt_eat + datetime.timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday or Sunday
        next_day = next_day + datetime.timedelta(days=1)
    
    # Set to 9 AM
    next_business_day = next_day.replace(
        hour=Config.BUSINESS_HOURS_START,
        minute=0,
        second=0,
        microsecond=0
    )
    
    return next_business_day