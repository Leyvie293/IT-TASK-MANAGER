# auth_controller.py - CORRECTED VERSION
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from functools import wraps
import re

# Import database and models
try:
    from models.database import db
    from models.user_models import User, ActivityLog, UserPreference
    from models.task_models import Task, TaskActivity
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.database import db
    from models.user_models import User, ActivityLog, UserPreference
    from models.task_models import Task, TaskActivity

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Initialize bcrypt
bcrypt = Bcrypt()

# Helper function to check if endpoint exists
def endpoint_exists(endpoint_name):
    """Check if an endpoint exists in the Flask app."""
    try:
        if current_app:
            return endpoint_name in current_app.view_functions
    except RuntimeError:
        pass
    return False

# Make endpoint_exists available to templates
auth_bp.app_context_processor(lambda: dict(endpoint_exists=endpoint_exists))

# Helper decorator for role-based access
def require_roles(*roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if current_user.role not in roles:
                flash('Access denied. Insufficient permissions.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    # If user is already logged in, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        if not email or not password:
            flash('Please enter both email and password', 'danger')
            return render_template('auth/login.html')
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Check if user is active
            if not user.is_active:
                flash('Your account has been deactivated. Please contact an administrator.', 'danger')
                return render_template('auth/login.html')
            
            # Verify password using bcrypt
            if bcrypt.check_password_hash(user.password_hash, password):
                login_user(user, remember=remember)
                
                # Update login stats
                user.last_login = datetime.utcnow()
                user.login_count = (user.login_count or 0) + 1
                
                # Log the login activity
                activity = ActivityLog(
                    user_id=user.id,
                    activity_type='User Login',
                    description=f'User logged in successfully',
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string,
                    status='success'
                )
                db.session.add(activity)
                db.session.commit()
                
                flash('Login successful!', 'success')
                
                # Redirect to next page or dashboard
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):
                    return redirect(next_page)
                return redirect(url_for('dashboard'))
            else:
                # Log failed login attempt
                activity = ActivityLog(
                    user_id=user.id if user else None,
                    activity_type='Failed Login',
                    description=f'Failed login attempt for email: {email}',
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string,
                    status='failed'
                )
                db.session.add(activity)
                db.session.commit()
                
                flash('Invalid email or password', 'danger')
        else:
            # Log failed login attempt for non-existent user
            activity = ActivityLog(
                user_id=None,
                activity_type='Failed Login',
                description=f'Failed login attempt for non-existent email: {email}',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                status='failed'
            )
            db.session.add(activity)
            db.session.commit()
            
            flash('Invalid email or password', 'danger')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    # Log the logout activity
    activity = ActivityLog(
        user_id=current_user.id,
        activity_type='User Logout',
        description=f'User logged out',
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string,
        status='success'
    )
    db.session.add(activity)
    
    username = current_user.full_name
    logout_user()
    
    db.session.commit()
    flash(f'{username} has been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/profile')
@login_required
def profile():
    """Display user profile - FIXED VERSION"""
    try:
        # Get user's recent activities
        recent_activities = ActivityLog.query.filter_by(user_id=current_user.id)\
            .order_by(ActivityLog.created_at.desc())\
            .limit(10).all()
        
        # Get user's task statistics - FIXED: Convert UUID to string for comparison
        user_id_str = str(current_user.id)
        total_tasks = Task.query.filter(Task.assigned_to == user_id_str).count()
        completed_tasks = Task.query.filter(
            Task.assigned_to == user_id_str,
            Task.status == 'Resolved'
        ).count()
        pending_tasks = Task.query.filter(
            Task.assigned_to == user_id_str,
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        
        # Calculate overdue tasks
        overdue_tasks = Task.query.filter(
            Task.due_date < datetime.utcnow(),
            Task.assigned_to == user_id_str,
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        
        return render_template('auth/profile.html', 
                             user=current_user,  # CRITICAL: Pass user variable
                             recent_activities=recent_activities,
                             total_tasks=total_tasks,
                             completed_tasks=completed_tasks,
                             pending_tasks=pending_tasks,
                             overdue_tasks=overdue_tasks)
    
    except Exception as e:
        flash(f'Error loading profile: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
@require_roles('Admin')
def register():
    """Register new users (admin only)"""
    if request.method == 'POST':
        try:
            # Get form data
            data = request.form
            
            # Required fields
            required_fields = ['employee_id', 'first_name', 'last_name', 'email', 
                             'password', 'confirm_password', 'role', 'department']
            
            for field in required_fields:
                if not data.get(field):
                    flash(f'{field.replace("_", " ").title()} is required', 'danger')
                    return redirect(url_for('auth.register'))
            
            # Validate email format
            email = data['email'].strip()
            if not re.match(r'^[a-zA-Z0-9._%+-]+@rileyfalcon\.com$', email):
                flash('Please use company email address (@rileyfalcon.com)', 'danger')
                return redirect(url_for('auth.register'))
            
            # Validate passwords match
            if data['password'] != data['confirm_password']:
                flash('Passwords do not match', 'danger')
                return redirect(url_for('auth.register'))
            
            # Check password strength
            if len(data['password']) < 8:
                flash('Password must be at least 8 characters long', 'danger')
                return redirect(url_for('auth.register'))
            
            # Check if user exists
            if User.query.filter_by(email=email).first():
                flash('User with this email already exists', 'danger')
                return redirect(url_for('auth.register'))
            
            if User.query.filter_by(employee_id=data['employee_id']).first():
                flash('Employee ID already exists', 'danger')
                return redirect(url_for('auth.register'))
            
            # Process skills
            skills = []
            if data.get('skills'):
                skills = [skill.strip() for skill in data['skills'].split(',') if skill.strip()]
            
            # Create new user
            hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
            
            user = User(
                employee_id=data['employee_id'].strip(),
                first_name=data['first_name'].strip(),
                last_name=data['last_name'].strip(),
                email=email,
                password_hash=hashed_password,
                role=data['role'],
                department=data['department'],
                job_title=data.get('job_title', '').strip(),
                phone=data.get('phone', '').strip(),
                skills=skills,
                is_active=True,
                is_available=True,
                max_tasks=int(data.get('max_tasks', 5))
            )
            
            db.session.add(user)
            
            # Create default preferences for the user
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
            
            # Log the user creation activity
            activity = ActivityLog(
                user_id=current_user.id,
                activity_type='User Created',
                description=f'Created new user: {user.full_name} ({user.email})',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                status='success'
            )
            db.session.add(activity)
            
            db.session.commit()
            
            flash(f'User {user.full_name} created successfully!', 'success')
            return redirect(url_for('auth.manage_users'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating user: {str(e)}', 'danger')
            return redirect(url_for('auth.register'))
    
    return render_template('auth/register.html')

@auth_bp.route('/users')
@login_required
@require_roles('Admin')
def manage_users():
    """Manage users (admin only)"""
    users = User.query.order_by(User.created_at.desc()).all()
    
    # Get statistics
    total_users = len(users)
    active_users = len([u for u in users if u.is_active])
    admin_users = len([u for u in users if u.role == 'Admin'])
    
    return render_template('auth/users.html', 
                         users=users,
                         total_users=total_users,
                         active_users=active_users,
                         admin_users=admin_users)

@auth_bp.route('/user/<user_id>')
@login_required
@require_roles('Admin')
def view_user(user_id):
    """View user details (admin only)"""
    user = User.query.get(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('auth.manage_users'))
    
    # Get user's recent activities
    recent_activities = ActivityLog.query.filter_by(user_id=user.id)\
        .order_by(ActivityLog.created_at.desc())\
        .limit(10).all()
    
    # Get user's task statistics
    user_id_str = str(user.id)  # Convert UUID to string
    total_tasks = Task.query.filter(Task.assigned_to == user_id_str).count()
    completed_tasks = Task.query.filter(
        Task.assigned_to == user_id_str,
        Task.status == 'Resolved'
    ).count()
    pending_tasks = Task.query.filter(
        Task.assigned_to == user_id_str,
        Task.status.in_(['New', 'Assigned', 'In Progress'])
    ).count()
    
    # Calculate overdue tasks
    overdue_tasks = Task.query.filter(
        Task.due_date < datetime.utcnow(),
        Task.assigned_to == user_id_str,
        Task.status.in_(['New', 'Assigned', 'In Progress'])
    ).count()
    
    return render_template('auth/view_user.html',
                         user=user,
                         recent_activities=recent_activities,
                         total_tasks=total_tasks,
                         completed_tasks=completed_tasks,
                         pending_tasks=pending_tasks,
                         overdue_tasks=overdue_tasks)

@auth_bp.route('/user/<user_id>/toggle', methods=['POST'])
@login_required
@require_roles('Admin')
def toggle_user_status(user_id):
    """Toggle user active status (admin only)"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Prevent deactivating yourself
    if user.id == current_user.id:
        return jsonify({'success': False, 'message': 'Cannot deactivate your own account'}), 400
    
    old_status = user.is_active
    user.is_active = not user.is_active
    new_status = user.is_active
    
    # Log the activity
    activity = ActivityLog(
        user_id=current_user.id,
        activity_type='User Status Changed',
        description=f'Changed user {user.full_name} status from {"Active" if old_status else "Inactive"} to {"Active" if new_status else "Inactive"}',
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string,
        status='success',
        old_value=f'Active: {old_status}',
        new_value=f'Active: {new_status}'
    )
    db.session.add(activity)
    
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    return jsonify({
        'success': True, 
        'message': f'User {status} successfully',
        'is_active': user.is_active
    })

@auth_bp.route('/user/<user_id>/update', methods=['POST'])
@login_required
@require_roles('Admin')
def update_user(user_id):
    """Update user information (admin only)"""
    user = User.query.get(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('auth.manage_users'))
    
    try:
        data = request.form
        
        # Track changes
        changes = []
        
        # Update fields
        if 'first_name' in data and data['first_name'] != user.first_name:
            changes.append(f"First name: {user.first_name} → {data['first_name']}")
            user.first_name = data['first_name'].strip()
        
        if 'last_name' in data and data['last_name'] != user.last_name:
            changes.append(f"Last name: {user.last_name} → {data['last_name']}")
            user.last_name = data['last_name'].strip()
        
        if 'email' in data and data['email'] != user.email:
            # Check if email is already taken
            existing_user = User.query.filter_by(email=data['email']).first()
            if existing_user and existing_user.id != user.id:
                flash('Email already in use by another user', 'danger')
                return redirect(url_for('auth.view_user', user_id=user.id))
            
            changes.append(f"Email: {user.email} → {data['email']}")
            user.email = data['email'].strip()
        
        if 'role' in data and data['role'] != user.role:
            changes.append(f"Role: {user.role} → {data['role']}")
            user.role = data['role']
        
        if 'department' in data and data['department'] != user.department:
            changes.append(f"Department: {user.department} → {data['department']}")
            user.department = data['department']
        
        if 'job_title' in data and data['job_title'] != user.job_title:
            changes.append(f"Job title: {user.job_title} → {data['job_title']}")
            user.job_title = data['job_title'].strip()
        
        if 'phone' in data and data['phone'] != user.phone:
            changes.append(f"Phone: {user.phone} → {data['phone']}")
            user.phone = data['phone'].strip()
        
        # Handle skills
        if 'skills' in data:
            skills = [skill.strip() for skill in data['skills'].split(',') if skill.strip()]
            old_skills = ', '.join(user.skills or [])
            new_skills = ', '.join(skills)
            if old_skills != new_skills:
                changes.append(f"Skills: {old_skills} → {new_skills}")
                user.skills = skills
        
        # Handle max tasks
        if 'max_tasks' in data:
            try:
                max_tasks = int(data['max_tasks'])
                if max_tasks != user.max_tasks:
                    changes.append(f"Max tasks: {user.max_tasks} → {max_tasks}")
                    user.max_tasks = max_tasks
            except ValueError:
                pass
        
        # Handle password reset if provided
        new_password = data.get('new_password', '').strip()
        if new_password:
            if len(new_password) < 8:
                flash('Password must be at least 8 characters long', 'danger')
                return redirect(url_for('auth.view_user', user_id=user.id))
            
            user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
            changes.append("Password: [RESET]")
        
        # Log the activity if there were changes
        if changes:
            activity = ActivityLog(
                user_id=current_user.id,
                activity_type='User Updated',
                description=f'Updated user {user.full_name}',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                status='success',
                old_value='',
                new_value=', '.join(changes)
            )
            db.session.add(activity)
        
        db.session.commit()
        
        if changes:
            flash('User updated successfully!', 'success')
        else:
            flash('No changes made', 'info')
        
        return redirect(url_for('auth.view_user', user_id=user.id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating user: {str(e)}', 'danger')
        return redirect(url_for('auth.view_user', user_id=user.id))

@auth_bp.route('/user/<user_id>/delete', methods=['POST'])
@login_required
@require_roles('Admin')
def delete_user(user_id):
    """Delete user (admin only)"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Prevent deleting yourself
    if user.id == current_user.id:
        return jsonify({'success': False, 'message': 'Cannot delete your own account'}), 400
    
    # Check if user has assigned tasks
    user_tasks = Task.query.filter(Task.assigned_to == str(user.id)).count()  # Convert to string
    if user_tasks > 0:
        return jsonify({
            'success': False, 
            'message': f'Cannot delete user with {user_tasks} assigned tasks. Reassign tasks first.'
        }), 400
    
    username = user.full_name
    
    # Log the deletion
    activity = ActivityLog(
        user_id=current_user.id,
        activity_type='User Deleted',
        description=f'Deleted user: {username} ({user.email})',
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string,
        status='success',
        old_value=str(user.id),
        new_value='DELETED'
    )
    db.session.add(activity)
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'success': True, 'message': f'User {username} deleted successfully'})

@auth_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Update user settings"""
    if request.method == 'POST':
        try:
            changes = []
            
            # Update basic info
            if 'phone' in request.form and request.form['phone'] != current_user.phone:
                changes.append(f"Phone: {current_user.phone} → {request.form['phone']}")
                current_user.phone = request.form['phone'].strip()
            
            if 'job_title' in request.form and request.form['job_title'] != current_user.job_title:
                changes.append(f"Job title: {current_user.job_title} → {request.form['job_title']}")
                current_user.job_title = request.form['job_title'].strip()
            
            # Handle skills
            if 'skills' in request.form:
                skills = [skill.strip() for skill in request.form['skills'].split(',') if skill.strip()]
                old_skills = ', '.join(current_user.skills or [])
                new_skills = ', '.join(skills)
                if old_skills != new_skills:
                    changes.append(f"Skills: {old_skills} → {new_skills}")
                    current_user.skills = skills
            
            # Handle password change
            current_password = request.form.get('current_password', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            if current_password or new_password or confirm_password:
                # All password fields must be filled
                if not all([current_password, new_password, confirm_password]):
                    flash('All password fields are required for password change', 'danger')
                    return redirect(url_for('auth.settings'))
                
                # Verify current password
                if not bcrypt.check_password_hash(current_user.password_hash, current_password):
                    flash('Current password is incorrect', 'danger')
                    return redirect(url_for('auth.settings'))
                
                # Check if new password matches confirmation
                if new_password != confirm_password:
                    flash('New passwords do not match', 'danger')
                    return redirect(url_for('auth.settings'))
                
                # Check password strength
                if len(new_password) < 8:
                    flash('New password must be at least 8 characters long', 'danger')
                    return redirect(url_for('auth.settings'))
                
                # Update password
                current_user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
                changes.append("Password: [CHANGED]")
                flash('Password updated successfully!', 'success')
            
            # Log the activity if there were changes
            if changes:
                activity = ActivityLog(
                    user_id=current_user.id,
                    activity_type='Settings Updated',
                    description=f'User updated their settings',
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string,
                    status='success',
                    old_value='',
                    new_value=', '.join(changes)
                )
                db.session.add(activity)
            
            db.session.commit()
            
            if changes:
                flash('Settings updated successfully!', 'success')
            else:
                flash('No changes made', 'info')
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating settings: {str(e)}', 'danger')
        
        return redirect(url_for('auth.settings'))
    
    return render_template('auth/settings.html')

# ====================================================================
# PROFILE MANAGEMENT ROUTES - FIXED VERSIONS
# ====================================================================

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    if request.method == 'POST':
        try:
            current_password = request.form.get('current_password', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            # Validation
            if not all([current_password, new_password, confirm_password]):
                flash('All password fields are required', 'danger')
                return redirect(url_for('auth.change_password'))
            
            # Verify current password
            if not bcrypt.check_password_hash(current_user.password_hash, current_password):
                flash('Current password is incorrect', 'danger')
                return redirect(url_for('auth.change_password'))
            
            # Check if new password matches confirmation
            if new_password != confirm_password:
                flash('New passwords do not match', 'danger')
                return redirect(url_for('auth.change_password'))
            
            # Check password strength
            if len(new_password) < 8:
                flash('New password must be at least 8 characters long', 'danger')
                return redirect(url_for('auth.change_password'))
            
            # Update password
            current_user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
            
            # Log the activity
            activity = ActivityLog(
                user_id=current_user.id,
                activity_type='Password Changed',
                description=f'User changed their password',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                status='success'
            )
            db.session.add(activity)
            db.session.commit()
            
            flash('Password changed successfully!', 'success')
            return redirect(url_for('auth.profile'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error changing password: {str(e)}', 'danger')
            return redirect(url_for('auth.change_password'))
    
    return render_template('auth/change_password.html')

@auth_bp.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    if request.method == 'POST':
        try:
            changes = []
            
            # Update basic info
            if 'first_name' in request.form and request.form['first_name'] != current_user.first_name:
                changes.append(f"First name: {current_user.first_name} → {request.form['first_name']}")
                current_user.first_name = request.form['first_name'].strip()
            
            if 'last_name' in request.form and request.form['last_name'] != current_user.last_name:
                changes.append(f"Last name: {current_user.last_name} → {request.form['last_name']}")
                current_user.last_name = request.form['last_name'].strip()
            
            if 'phone' in request.form and request.form['phone'] != current_user.phone:
                changes.append(f"Phone: {current_user.phone} → {request.form['phone']}")
                current_user.phone = request.form['phone'].strip()
            
            if 'job_title' in request.form and request.form['job_title'] != current_user.job_title:
                changes.append(f"Job title: {current_user.job_title} → {request.form['job_title']}")
                current_user.job_title = request.form['job_title'].strip()
            
            # Handle skills
            if 'skills' in request.form:
                skills = [skill.strip() for skill in request.form['skills'].split(',') if skill.strip()]
                old_skills = ', '.join(current_user.skills or [])
                new_skills = ', '.join(skills)
                if old_skills != new_skills:
                    changes.append(f"Skills: {old_skills} → {new_skills}")
                    current_user.skills = skills
            
            # Log the activity if there were changes
            if changes:
                activity = ActivityLog(
                    user_id=current_user.id,
                    activity_type='Profile Updated',
                    description=f'User updated their profile',
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string,
                    status='success',
                    old_value='',
                    new_value=', '.join(changes)
                )
                db.session.add(activity)
            
            db.session.commit()
            
            if changes:
                flash('Profile updated successfully!', 'success')
            else:
                flash('No changes made', 'info')
            
            return redirect(url_for('auth.profile'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'danger')
            return redirect(url_for('auth.edit_profile'))
    
    return render_template('auth/edit_profile.html', user=current_user)

@auth_bp.route('/activity-log')
@login_required
def activity_log():
    """View user activity log"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    activities = ActivityLog.query.filter_by(user_id=current_user.id)\
        .order_by(ActivityLog.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('auth/activity_log.html', activities=activities)

@auth_bp.route('/upload-profile-picture', methods=['POST'])
@login_required
def upload_profile_picture():
    """Upload profile picture (placeholder - returns error)"""
    flash('Profile picture upload is not yet implemented', 'warning')
    return redirect(url_for('auth.profile'))

# Simple routes for now to prevent redirect loops
@auth_bp.route('/')
def auth_index():
    """Auth blueprint index"""
    return redirect(url_for('auth.login'))

# Error handlers
@auth_bp.errorhandler(404)
def not_found(error):
    return render_template('auth/404.html'), 404

@auth_bp.errorhandler(403)
def forbidden(error):
    return render_template('auth/403.html'), 403

@auth_bp.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    flash('An internal error occurred. Please try again.', 'danger')
    return redirect(url_for('auth.login'))

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_user_statistics(user_id):
    """Get comprehensive statistics for a user"""
    user = User.query.get(user_id)
    if not user:
        return None
    
    user_id_str = str(user_id)
    
    # Task statistics
    total_tasks = Task.query.filter(Task.assigned_to == user_id_str).count()
    completed_tasks = Task.query.filter(
        Task.assigned_to == user_id_str,
        Task.status == 'Resolved'
    ).count()
    pending_tasks = Task.query.filter(
        Task.assigned_to == user_id_str,
        Task.status.in_(['New', 'Assigned', 'In Progress'])
    ).count()
    overdue_tasks = Task.query.filter(
        Task.due_date < datetime.utcnow(),
        Task.assigned_to == user_id_str,
        Task.status.in_(['New', 'Assigned', 'In Progress'])
    ).count()
    
    # Calculate completion rate
    completion_rate = round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1)
    
    # Get today's tasks
    today = datetime.utcnow().date()
    tasks_today = Task.query.filter(
        Task.assigned_to == user_id_str,
        db.func.date(Task.created_at) == today
    ).count()
    
    tasks_completed_today = Task.query.filter(
        Task.assigned_to == user_id_str,
        Task.status == 'Resolved',
        db.func.date(Task.completed_at) == today
    ).count()
    
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'pending_tasks': pending_tasks,
        'overdue_tasks': overdue_tasks,
        'completion_rate': completion_rate,
        'tasks_today': tasks_today,
        'tasks_completed_today': tasks_completed_today
    }

def log_activity(user_id, activity_type, description, status='success', **kwargs):
    """Helper function to log user activities"""
    try:
        activity = ActivityLog(
            user_id=user_id,
            activity_type=activity_type,
            description=description,
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            status=status,
            old_value=kwargs.get('old_value'),
            new_value=kwargs.get('new_value'),
            additional_data=kwargs.get('additional_data', {})
        )
        db.session.add(activity)
        db.session.commit()
        return activity
    except Exception as e:
        print(f"Error logging activity: {e}")
        return None