"""
Validation utilities for the I.T Task Manager
"""

import re
from datetime import datetime, timedelta
from flask import current_app

def validate_email(email):
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    # Check if it's a company email (optional)
    if current_app.config.get('REQUIRE_COMPANY_EMAIL', False):
        if not email.endswith('@rileyfalcon.com'):
            return False, "Please use your company email address (@rileyfalcon.com)"
    
    return True, ""

def validate_phone(phone):
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not phone:
        return True, ""  # Phone is optional
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a valid length (10 digits for US/Canada)
    if len(digits) != 10:
        return False, "Phone number must be 10 digits"
    
    return True, ""

def validate_password(password):
    """
    Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple (is_valid, error_message, strength_score)
    """
    if not password:
        return False, "Password is required", 0
    
    errors = []
    score = 0
    
    # Check length
    if len(password) >= 8:
        score += 1
    else:
        errors.append("Password must be at least 8 characters long")
    
    # Check for uppercase letters
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letters
    if re.search(r'[a-z]', password):
        score += 1
    else:
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for numbers
    if re.search(r'\d', password):
        score += 1
    else:
        errors.append("Password must contain at least one number")
    
    # Check for special characters
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    else:
        errors.append("Password must contain at least one special character")
    
    # Determine if valid based on minimum requirements
    is_valid = len(password) >= 8 and score >= 3  # At least 3 out of 5 criteria
    
    error_message = "; ".join(errors) if not is_valid else ""
    
    return is_valid, error_message, score

def validate_task_data(task_data):
    """
    Validate task creation/update data
    
    Args:
        task_data: Dictionary containing task data
        
    Returns:
        Tuple (is_valid, error_message, validated_data)
    """
    errors = []
    validated = {}
    
    # Required fields
    required_fields = ['title', 'category', 'priority', 'department']
    
    for field in required_fields:
        if field not in task_data or not task_data[field]:
            errors.append(f"{field.replace('_', ' ').title()} is required")
        else:
            validated[field] = task_data[field].strip()
    
    # Validate title length
    if 'title' in validated and len(validated['title']) > 200:
        errors.append("Title must be less than 200 characters")
    
    # Validate description (optional but if provided, check length)
    if 'description' in task_data and task_data['description']:
        description = task_data['description'].strip()
        if len(description) > 5000:
            errors.append("Description must be less than 5000 characters")
        validated['description'] = description
    
    # Validate priority
    if 'priority' in validated:
        valid_priorities = ['Low', 'Medium', 'High', 'Critical']
        if validated['priority'] not in valid_priorities:
            errors.append(f"Priority must be one of: {', '.join(valid_priorities)}")
    
    # Validate category
    if 'category' in validated:
        valid_categories = current_app.config.get('TASK_CATEGORIES', [])
        if validated['category'] not in valid_categories:
            errors.append(f"Category must be one of: {', '.join(valid_categories)}")
    
    # Validate department
    if 'department' in validated:
        valid_departments = current_app.config.get('DEPARTMENTS', [])
        if validated['department'] not in valid_departments:
            errors.append(f"Department must be one of: {', '.join(valid_departments)}")
    
    # Validate due date (if provided)
    if 'due_date' in task_data and task_data['due_date']:
        try:
            due_date = datetime.strptime(task_data['due_date'], '%Y-%m-%d')
            if due_date.date() < datetime.now().date():
                errors.append("Due date cannot be in the past")
            validated['due_date'] = due_date
        except ValueError:
            errors.append("Invalid due date format. Use YYYY-MM-DD")
    
    # Validate estimated hours (if provided)
    if 'estimated_hours' in task_data and task_data['estimated_hours']:
        try:
            hours = float(task_data['estimated_hours'])
            if hours <= 0 or hours > 1000:
                errors.append("Estimated hours must be between 0.1 and 1000")
            validated['estimated_hours'] = hours
        except ValueError:
            errors.append("Estimated hours must be a number")
    
    # Validate location (if provided)
    if 'location' in task_data and task_data['location']:
        location = task_data['location'].strip()
        if len(location) > 200:
            errors.append("Location must be less than 200 characters")
        validated['location'] = location
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message, validated

def validate_user_data(user_data, is_update=False):
    """
    Validate user registration/update data
    
    Args:
        user_data: Dictionary containing user data
        is_update: Whether this is an update (some fields optional)
        
    Returns:
        Tuple (is_valid, error_message, validated_data)
    """
    errors = []
    validated = {}
    
    # Required fields for registration (optional for update)
    required_for_create = ['employee_id', 'first_name', 'last_name', 'email', 
                          'password', 'role', 'department']
    
    for field in required_for_create:
        if not is_update and (field not in user_data or not user_data[field]):
            errors.append(f"{field.replace('_', ' ').title()} is required")
        elif field in user_data and user_data[field]:
            validated[field] = user_data[field].strip()
    
    # Validate email
    if 'email' in validated:
        is_valid_email, email_error = validate_email(validated['email'])
        if not is_valid_email:
            errors.append(email_error)
    
    # Validate password (only for creation or password change)
    if 'password' in validated and not is_update:
        is_valid_pw, pw_error, _ = validate_password(validated['password'])
        if not is_valid_pw:
            errors.append(pw_error)
    
    # Validate phone (optional)
    if 'phone' in user_data and user_data['phone']:
        is_valid_phone, phone_error = validate_phone(user_data['phone'])
        if not is_valid_phone:
            errors.append(phone_error)
        else:
            validated['phone'] = user_data['phone'].strip()
    
    # Validate role
    if 'role' in validated:
        valid_roles = ['Admin', 'Supervisor', 'Technician', 'Requester']
        if validated['role'] not in valid_roles:
            errors.append(f"Role must be one of: {', '.join(valid_roles)}")
    
    # Validate department
    if 'department' in validated:
        valid_departments = current_app.config.get('DEPARTMENTS', [])
        if validated['department'] not in valid_departments:
            errors.append(f"Department must be one of: {', '.join(valid_departments)}")
    
    # Validate job title (optional)
    if 'job_title' in user_data and user_data['job_title']:
        title = user_data['job_title'].strip()
        if len(title) > 100:
            errors.append("Job title must be less than 100 characters")
        validated['job_title'] = title
    
    # Validate skills (optional)
    if 'skills' in user_data and user_data['skills']:
        if isinstance(user_data['skills'], str):
            skills = [s.strip() for s in user_data['skills'].split(',')]
        else:
            skills = user_data['skills']
        
        if not isinstance(skills, list):
            errors.append("Skills must be a list or comma-separated string")
        else:
            validated['skills'] = skills
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message, validated

def validate_report_parameters(params):
    """
    Validate report generation parameters
    
    Args:
        params: Dictionary containing report parameters
        
    Returns:
        Tuple (is_valid, error_message, validated_params)
    """
    errors = []
    validated = {}
    
    # Required fields
    required_fields = ['report_type', 'start_date', 'end_date', 'format']
    
    for field in required_fields:
        if field not in params or not params[field]:
            errors.append(f"{field.replace('_', ' ').title()} is required")
        else:
            validated[field] = params[field]
    
    # Validate report type
    if 'report_type' in validated:
        valid_types = ['sla', 'performance', 'department', 'technician', 
                      'category', 'trends', 'custom']
        if validated['report_type'] not in valid_types:
            errors.append(f"Report type must be one of: {', '.join(valid_types)}")
    
    # Validate dates
    for date_field in ['start_date', 'end_date']:
        if date_field in validated:
            try:
                date_obj = datetime.strptime(validated[date_field], '%Y-%m-%d')
                validated[f'{date_field}_obj'] = date_obj
            except ValueError:
                errors.append(f"Invalid {date_field.replace('_', ' ')} format. Use YYYY-MM-DD")
    
    # Validate date range
    if 'start_date_obj' in validated and 'end_date_obj' in validated:
        if validated['end_date_obj'] < validated['start_date_obj']:
            errors.append("End date must be after start date")
        
        # Check if date range is too large
        date_range = (validated['end_date_obj'] - validated['start_date_obj']).days
        if date_range > 365:
            errors.append("Date range cannot exceed 1 year for performance reasons")
    
    # Validate format
    if 'format' in validated:
        valid_formats = ['pdf', 'excel', 'csv']
        if validated['format'] not in valid_formats:
            errors.append(f"Format must be one of: {', '.join(valid_formats)}")
    
    # Validate department (optional)
    if 'department' in params and params['department']:
        valid_departments = current_app.config.get('DEPARTMENTS', [])
        if params['department'] not in valid_departments:
            errors.append(f"Department must be one of: {', '.join(valid_departments)}")
        else:
            validated['department'] = params['department']
    
    # Validate technician_id (if provided for technician report)
    if 'technician_id' in params and params['technician_id']:
        if validated.get('report_type') == 'technician':
            validated['technician_id'] = params['technician_id']
        else:
            errors.append("Technician ID is only valid for technician reports")
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message, validated