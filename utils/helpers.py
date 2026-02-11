"""
Helper functions for the I.T Task Manager
"""

import re
from datetime import datetime, timedelta
from flask import current_app
import random
import string
import json

def format_date(date_obj, format_type='full'):
    """
    Format date object to string
    
    Args:
        date_obj: Date/datetime object
        format_type: 'full', 'date', 'time', 'relative', 'short'
        
    Returns:
        Formatted date string
    """
    if not date_obj:
        return "N/A"
    
    if format_type == 'full':
        return date_obj.strftime('%Y-%m-%d %H:%M:%S')
    elif format_type == 'date':
        return date_obj.strftime('%Y-%m-%d')
    elif format_type == 'time':
        return date_obj.strftime('%H:%M:%S')
    elif format_type == 'short':
        return date_obj.strftime('%m/%d/%y %H:%M')
    elif format_type == 'relative':
        return get_relative_time(date_obj)
    else:
        return str(date_obj)

def get_relative_time(date_obj):
    """
    Get human-readable relative time (e.g., "2 hours ago")
    
    Args:
        date_obj: Date/datetime object
        
    Returns:
        Relative time string
    """
    now = datetime.utcnow()
    diff = now - date_obj
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def format_duration(seconds, format_type='auto'):
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        format_type: 'auto', 'hours', 'minutes', 'detailed'
        
    Returns:
        Formatted duration string
    """
    if seconds is None:
        return "N/A"
    
    if format_type == 'auto':
        if seconds >= 3600:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        elif seconds >= 60:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            return f"{seconds:.0f}s"
    
    elif format_type == 'hours':
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    
    elif format_type == 'minutes':
        minutes = seconds / 60
        return f"{minutes:.0f} minutes"
    
    elif format_type == 'detailed':
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    return str(seconds)

def generate_task_id():
    """
    Generate unique task ID
    
    Returns:
        Task ID string (e.g., IT-2024-0001)
    """
    from models.task_models import Task
    
    year = datetime.now().year
    
    # Get the last task ID
    last_task = Task.query.order_by(Task.created_at.desc()).first()
    
    if last_task and last_task.task_id:
        # Extract sequence number
        match = re.search(r'IT-\d{4}-(\d{4})', last_task.task_id)
        if match:
            last_number = int(match.group(1))
            new_number = last_number + 1
        else:
            new_number = 1
    else:
        new_number = 1
    
    return f"IT-{year}-{new_number:04d}"

def calculate_sla_due_date(priority, created_date=None):
    """
    Calculate SLA due date based on priority
    
    Args:
        priority: Task priority (Critical, High, Medium, Low)
        created_date: Task creation date (defaults to now)
        
    Returns:
        SLA due date
    """
    if not created_date:
        created_date = datetime.utcnow()
    
    sla_hours = {
        'Critical': current_app.config.get('SLA_CRITICAL', 2),
        'High': current_app.config.get('SLA_HIGH', 4),
        'Medium': current_app.config.get('SLA_MEDIUM', 24),
        'Low': current_app.config.get('SLA_LOW', 72)
    }
    
    hours = sla_hours.get(priority, 24)
    return created_date + timedelta(hours=hours)

def get_sla_status(task):
    """
    Get SLA status for a task
    
    Args:
        task: Task object
        
    Returns:
        SLA status string and color
    """
    if not task.sla_due_date:
        return 'No SLA', 'secondary'
    
    now = datetime.utcnow()
    
    if task.status in ['Closed', 'Resolved']:
        if task.end_time and task.end_time <= task.sla_due_date:
            return 'Met', 'success'
        else:
            return 'Breached', 'danger'
    
    # For open tasks
    time_remaining = task.sla_due_date - now
    hours_remaining = time_remaining.total_seconds() / 3600
    
    if hours_remaining < 0:
        return 'Breached', 'danger'
    elif hours_remaining < 2:
        return 'At Risk', 'warning'
    else:
        return 'Within SLA', 'success'

def truncate_text(text, max_length=100, suffix='...'):
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def generate_random_string(length=8, include_digits=True, include_special=False):
    """
    Generate random string
    
    Args:
        length: Length of string
        include_digits: Include digits
        include_special: Include special characters
        
    Returns:
        Random string
    """
    characters = string.ascii_letters
    
    if include_digits:
        characters += string.digits
    
    if include_special:
        characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    return ''.join(random.choice(characters) for _ in range(length))

def safe_json_loads(data, default=None):
    """
    Safely load JSON data
    
    Args:
        data: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed data or default
    """
    if default is None:
        default = {}
    
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def dict_to_query_string(data):
    """
    Convert dictionary to query string
    
    Args:
        data: Dictionary
        
    Returns:
        Query string
    """
    return '&'.join([f"{k}={v}" for k, v in data.items() if v is not None])

def get_pagination_info(page, per_page, total):
    """
    Calculate pagination information
    
    Args:
        page: Current page
        per_page: Items per page
        total: Total items
        
    Returns:
        Dictionary with pagination info
    """
    total_pages = (total + per_page - 1) // per_page
    
    return {
        'current_page': page,
        'per_page': per_page,
        'total_items': total,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None
    }

def calculate_progress(checklist):
    """
    Calculate progress percentage from checklist
    
    Args:
        checklist: List of checklist items with 'completed' field
        
    Returns:
        Progress percentage (0-100)
    """
    if not checklist:
        return 0
    
    completed = sum(1 for item in checklist if item.get('completed', False))
    return int((completed / len(checklist)) * 100)

def get_priority_color(priority):
    """
    Get Bootstrap color class for priority
    
    Args:
        priority: Priority level
        
    Returns:
        Bootstrap color class
    """
    colors = {
        'Critical': 'danger',
        'High': 'warning',
        'Medium': 'info',
        'Low': 'success'
    }
    
    return colors.get(priority, 'secondary')

def get_status_color(status):
    """
    Get Bootstrap color class for status
    
    Args:
        status: Task status
        
    Returns:
        Bootstrap color class
    """
    colors = {
        'New': 'primary',
        'Acknowledged': 'info',
        'Assigned': 'secondary',
        'In Progress': 'warning',
        'Waiting': 'dark',
        'Escalated': 'danger',
        'Resolved': 'success',
        'Closed': 'success'
    }
    
    return colors.get(status, 'secondary')

def format_file_size(bytes_size):
    """
    Format file size in human-readable format
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"

def validate_file_extension(filename, allowed_extensions=None):
    """
    Validate file extension
    
    Args:
        filename: Name of file
        allowed_extensions: List of allowed extensions
        
    Returns:
        Boolean indicating if extension is valid
    """
    if allowed_extensions is None:
        allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', 
                                                   {'png', 'jpg', 'jpeg', 'gif', 'pdf'})
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions