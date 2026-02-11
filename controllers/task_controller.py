# controllers/task_controller.py - COMPLETE FIXED VERSION WITH CATEGORY
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, send_file, abort
from flask_login import login_required, current_user
from datetime import datetime, timedelta
import json
from functools import wraps
import csv
import io
from models.database import db

task_bp = Blueprint('task', __name__, url_prefix='/tasks')

# ====================================================================
# PERMISSION CHECK FUNCTION
# ====================================================================
def has_task_access(task, user):
    """Check if user has access to view/edit a task based on role"""
    if not task:
        return False
    
    if user.role == 'Admin':
        return True
    elif user.role == 'Supervisor':
        return task.department == user.department
    else:
        # FIXED: Convert both to strings for comparison
        return str(task.assigned_to) == str(user.id)

def admin_or_supervisor_required(f):
    """Decorator to require admin or supervisor role"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role not in ['Admin', 'Supervisor']:
            flash('Administrator or supervisor access required.', 'danger')
            return redirect(url_for('task.task_list'))
        return f(*args, **kwargs)
    return decorated_function

# ====================================================================
# MAIN TASK LIST ROUTE - FIXED WITH DATETIME ISSUE RESOLVED
# ====================================================================

@task_bp.route('/')
@task_bp.route('/list')
@login_required
def task_list():
    """View tasks with role-based filtering"""
    from models.task_models import Task
    from models.user_models import User
    
    # Get current time - FIXED: Make it offset-naive for consistent comparison
    now = datetime.utcnow().replace(tzinfo=None, microsecond=0)
    
    # Get filter parameters
    status = request.args.get('status', 'all')
    priority = request.args.get('priority', 'all')
    assigned_to = request.args.get('assigned_to', 'all')
    department = request.args.get('department', 'all')
    category = request.args.get('category', 'all')
    filter_type = request.args.get('filter', 'all')
    search = request.args.get('q', '').strip()
    sort = request.args.get('sort', 'created_at_desc')
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # ROLE-BASED QUERY FILTERING
    if current_user.role == 'Admin':
        # Admin sees ALL tasks across ALL departments
        query = Task.query
        page_title = "All Tasks (All Departments)"
        is_admin_view = True
        
    elif current_user.role == 'Supervisor':
        # Supervisor sees tasks ONLY in their department
        if not current_user.department:
            flash('You are not assigned to any department.', 'warning')
            return redirect(url_for('dashboard'))
        
        query = Task.query.filter_by(department=current_user.department)
        page_title = f"All Tasks ({current_user.department} Department)"
        is_admin_view = False
        
    else:
        # Regular users see ONLY tasks assigned to them
        user_id_str = str(current_user.id)
        query = Task.query.filter(Task.assigned_to == user_id_str)
        page_title = "My Tasks"
        is_admin_view = False
    
    # Apply quick filters
    if filter_type == 'my_tasks':
        user_id_str = str(current_user.id)
        query = query.filter_by(assigned_to=user_id_str)
    elif filter_type == 'overdue':
        # FIXED: Use offset-naive datetime for comparison
        query = query.filter(
            Task.due_date < datetime.utcnow().replace(tzinfo=None),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        )
    elif filter_type == 'unassigned':
        query = query.filter(Task.assigned_to.is_(None))
    elif filter_type == 'high_priority':
        query = query.filter_by(priority='High')
    elif filter_type == 'critical':
        query = query.filter_by(priority='Critical')
    elif filter_type == 'my_department' and current_user.role == 'Supervisor':
        query = query.filter_by(department=current_user.department)
    
    # Apply advanced filters
    if status and status != 'all':
        query = query.filter_by(status=status)
    
    if priority and priority != 'all':
        query = query.filter_by(priority=priority)
    
    if assigned_to:
        if assigned_to == 'me':
            user_id_str = str(current_user.id)
            query = query.filter_by(assigned_to=user_id_str)
        elif assigned_to == 'unassigned':
            query = query.filter(Task.assigned_to.is_(None))
        elif assigned_to != 'all':
            query = query.filter_by(assigned_to=assigned_to)
    
    if department and department != 'all' and current_user.role == 'Admin':
        query = query.filter_by(department=department)
    
    if category and category != 'all':
        query = query.filter_by(category=category)
    
    # Apply search
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            db.or_(
                Task.title.ilike(search_term),
                Task.description.ilike(search_term),
                Task.id.ilike(search_term) if search.isdigit() else False,
                Task.category.ilike(search_term)
            )
        )
    
    # Apply sorting
    if sort == 'due_date_asc':
        query = query.order_by(Task.due_date.asc())
    elif sort == 'due_date_desc':
        query = query.order_by(Task.due_date.desc())
    elif sort == 'priority_desc':
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        query = query.order_by(db.case(priority_order, value=Task.priority))
    elif sort == 'status':
        query = query.order_by(Task.status)
    elif sort == 'created_at_desc':
        query = query.order_by(Task.created_at.desc())
    elif sort == 'created_at_asc':
        query = query.order_by(Task.created_at.asc())
    else:
        query = query.order_by(Task.created_at.desc())
    
    # Paginate results
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    tasks = pagination.items
    
    # FIXED: Ensure all task due dates are offset-naive for template comparison
    for task in tasks:
        if task.due_date and task.due_date.tzinfo is not None:
            task.due_date = task.due_date.replace(tzinfo=None)
    
    # Get users for display and filter dropdown
    if current_user.role == 'Admin':
        users = User.query.filter_by(is_active=True).order_by(User.first_name, User.last_name).all()
    elif current_user.role == 'Supervisor':
        users = User.query.filter_by(
            department=current_user.department,
            is_active=True
        ).order_by(User.first_name, User.last_name).all()
    else:
        users = User.query.filter_by(id=current_user.id, is_active=True).all()
    
    # Get unique departments for filter dropdown (Admin only)
    departments = db.session.query(Task.department).distinct().all()
    department_list = [dept[0] for dept in departments if dept[0]]
    
    # Get unique categories for filter dropdown
    categories = db.session.query(Task.category).distinct().all()
    category_list = [cat[0] for cat in categories if cat[0]]
    
    # Helper function for template
    def get_department_users():
        if current_user.role == 'Admin':
            return User.query.filter_by(is_active=True).all()
        elif current_user.role == 'Supervisor':
            return User.query.filter_by(
                department=current_user.department,
                is_active=True
            ).all()
        else:
            return [current_user]
    
    # Helper function for template
    def can_close_tasks():
        return current_user.role in ['Admin', 'Supervisor']
    
    return render_template('tasks/list.html',
                         tasks=tasks,
                         pagination=pagination,
                         users=users,
                         departments=department_list,
                         categories=category_list,
                         page_title=page_title,
                         now=now,
                         is_admin_view=is_admin_view,
                         current_status=status,
                         current_priority=priority,
                         current_assigned=assigned_to,
                         current_department=department,
                         current_category=category,
                         current_filter=filter_type,
                         current_search=search,
                         current_sort=sort,
                         has_task_access=has_task_access,
                         get_department_users=get_department_users,
                         can_close_tasks=can_close_tasks)

# ====================================================================
# TASK DETAIL VIEW
# ====================================================================

@task_bp.route('/<task_id>')
@login_required
def view(task_id):
    """View a single task"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access based on role
    if not has_task_access(task, current_user):
        flash('Access denied. You do not have permission to view this task.', 'danger')
        return redirect(url_for('task.task_list'))
    
    # FIXED: Ensure task due_date is offset-naive for template
    if task.due_date and task.due_date.tzinfo is not None:
        task.due_date = task.due_date.replace(tzinfo=None)
    
    # Get assigned user details
    assigned_user = None
    if task.assigned_to:
        assigned_user = User.query.get(str(task.assigned_to))
    
    # Get created by user details
    created_by_user = None
    if task.created_by:
        created_by_user = User.query.get(str(task.created_by))
    
    # FIXED: Pass 'now' to template for comparison
    now = datetime.utcnow().replace(tzinfo=None, microsecond=0)
    
    return render_template('tasks/detail.html',
                         task=task,
                         assigned_user=assigned_user,
                         created_by_user=created_by_user,
                         now=now,
                         has_task_access=has_task_access)

# ====================================================================
# CREATE TASK - FIXED WITH CATEGORY
# ====================================================================

@task_bp.route('/create', methods=['GET', 'POST'])
@admin_or_supervisor_required
def create():
    """Create a new task"""
    from models.task_models import Task
    from models.user_models import User
    
    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title', '').strip()
            description = request.form.get('description', '').strip()
            category = request.form.get('category', '').strip()
            status = request.form.get('status', 'New')
            priority = request.form.get('priority', 'Medium')
            department = request.form.get('department', '').strip()
            assigned_to = request.form.get('assigned_to', '').strip()
            due_date = request.form.get('due_date', '').strip()
            estimated_hours = request.form.get('estimated_hours', 0)
            
            # Validate required fields
            if not all([title, description, category]):
                flash('Please fill in all required fields', 'danger')
                return redirect(url_for('task.create'))
            
            # Department handling
            if not department:
                if current_user.role == 'Admin':
                    flash('Department is required', 'danger')
                    return redirect(url_for('task.create'))
                else:
                    department = current_user.department
                    if not department:
                        flash('You are not assigned to any department. Please contact an administrator.', 'danger')
                        return redirect(url_for('task.create'))
            
            # Create task
            task = Task(
                title=title,
                description=description,
                category=category,
                status=status,
                priority=priority,
                department=department,
                created_by=str(current_user.id)
            )
            
            if assigned_to:
                try:
                    user = User.query.get(str(assigned_to))
                    if user:
                        # Validate department for supervisors
                        if current_user.role == 'Supervisor' and user.department != current_user.department:
                            flash('Can only assign to users in your department', 'danger')
                            return redirect(url_for('task.create'))
                        
                        task.assigned_to = user.id
                        task.assigned_by = str(current_user.id)
                        task.assigned_at = datetime.utcnow()
                except (ValueError, AttributeError):
                    pass
            
            if due_date:
                try:
                    task.due_date = datetime.strptime(due_date, '%Y-%m-%dT%H:%M')
                except ValueError:
                    try:
                        task.due_date = datetime.strptime(due_date, '%Y-%m-%d')
                    except ValueError:
                        pass
            
            if estimated_hours:
                try:
                    task.estimated_hours = float(estimated_hours)
                except ValueError:
                    task.estimated_hours = 0
            
            db.session.add(task)
            db.session.commit()
            
            flash(f'Task "{title}" created successfully!', 'success')
            return redirect(url_for('task.view', task_id=task.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating task: {str(e)}', 'danger')
            return redirect(url_for('task.create'))
    
    # GET request - show create form
    if current_user.role == 'Admin':
        users = User.query.filter_by(is_active=True).all()
        departments = db.session.query(User.department).distinct().all()
        departments = [dept[0] for dept in departments if dept[0]]
    else:
        users = User.query.filter_by(
            department=current_user.department,
            is_active=True
        ).all()
        departments = [current_user.department]
    
    # Get existing categories for dropdown
    from models.task_models import Task
    existing_categories = db.session.query(Task.category).distinct().all()
    categories = [cat[0] for cat in existing_categories if cat[0]]
    
    # Default categories if none exist
    if not categories:
        categories = [
            'IT Support',
            'Hardware',
            'Software',
            'Network',
            'Security',
            'Database',
            'General',
            'Infrastructure',
            'Help Desk',
            'Project'
        ]
    
    # FIXED: Pass 'now' to template for default due date
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    return render_template('tasks/create.html',
                         users=users,
                         departments=departments,
                         categories=categories,
                         tomorrow=tomorrow,
                         has_task_access=has_task_access)

# ====================================================================
# EDIT TASK - UPDATED WITH CATEGORY VALIDATION
# ====================================================================

@task_bp.route('/<task_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(task_id):
    """Edit an existing task"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access based on role
    if not has_task_access(task, current_user):
        flash('Access denied. You do not have permission to edit this task.', 'danger')
        return redirect(url_for('task.task_list'))
    
    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title', '').strip()
            description = request.form.get('description', '').strip()
            category = request.form.get('category', '').strip()
            status = request.form.get('status', task.status)
            priority = request.form.get('priority', task.priority)
            department = request.form.get('department', '').strip()
            assigned_to = request.form.get('assigned_to', '').strip()
            due_date = request.form.get('due_date', '').strip()
            progress = request.form.get('progress', 0)
            notes = request.form.get('notes', '').strip()
            estimated_hours = request.form.get('estimated_hours', 0)
            
            # Validate required fields
            if not all([title, description, category]):
                flash('Please fill in all required fields', 'danger')
                return redirect(url_for('task.edit', task_id=task_id))
            
            # Prevent non-admins from closing tasks
            if status in ['Resolved', 'Closed'] and current_user.role != 'Admin':
                flash('Only administrators can close tasks. Please contact an admin.', 'danger')
                return redirect(url_for('task.edit', task_id=task_id))
            
            # Update task
            task.title = title
            task.description = description
            task.category = category
            task.status = status
            task.priority = priority
            task.progress = int(progress) if progress.isdigit() and 0 <= int(progress) <= 100 else task.progress
            
            # Department - only admins can change
            if department and current_user.role == 'Admin':
                task.department = department
            
            if assigned_to:
                try:
                    user = User.query.get(str(assigned_to))
                    if user:
                        # Validate department for supervisors
                        if current_user.role == 'Supervisor' and user.department != current_user.department:
                            flash('Can only assign to users in your department', 'danger')
                            return redirect(url_for('task.edit', task_id=task_id))
                        
                        task.assigned_to = user.id
                        task.assigned_by = str(current_user.id)
                        task.assigned_at = datetime.utcnow()
                except (ValueError, AttributeError):
                    pass
            elif assigned_to == '':
                # Clear assignment
                task.assigned_to = None
                task.assigned_by = None
                task.assigned_at = None
            
            if due_date:
                try:
                    task.due_date = datetime.strptime(due_date, '%Y-%m-%dT%H:%M')
                except ValueError:
                    try:
                        task.due_date = datetime.strptime(due_date, '%Y-%m-%d')
                    except ValueError:
                        pass
            else:
                task.due_date = None
            
            if estimated_hours:
                try:
                    task.estimated_hours = float(estimated_hours)
                except ValueError:
                    pass
            
            task.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            flash(f'Task "{title}" updated successfully!', 'success')
            return redirect(url_for('task.view', task_id=task.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating task: {str(e)}', 'danger')
            return redirect(url_for('task.edit', task_id=task_id))
    
    # GET request - show edit form
    if current_user.role == 'Admin':
        users = User.query.filter_by(is_active=True).all()
        departments = db.session.query(User.department).distinct().all()
        departments = [dept[0] for dept in departments if dept[0]]
    elif current_user.role == 'Supervisor':
        users = User.query.filter_by(
            department=current_user.department,
            is_active=True
        ).all()
        departments = [current_user.department]
    else:
        users = User.query.filter_by(id=current_user.id, is_active=True).all()
        departments = [current_user.department] if current_user.department else []
    
    # Get existing categories for dropdown
    from models.task_models import Task
    existing_categories = db.session.query(Task.category).distinct().all()
    categories = [cat[0] for cat in existing_categories if cat[0]]
    
    # Default categories if none exist
    if not categories:
        categories = [
            'IT Support',
            'Hardware',
            'Software',
            'Network',
            'Security',
            'Database',
            'General',
            'Infrastructure',
            'Help Desk',
            'Project'
        ]
    
    # FIXED: Ensure task due_date is offset-naive for template
    if task.due_date and task.due_date.tzinfo is not None:
        task.due_date = task.due_date.replace(tzinfo=None)
    
    return render_template('tasks/edit.html',
                         task=task,
                         users=users,
                         departments=departments,
                         categories=categories,
                         has_task_access=has_task_access)

# ====================================================================
# DELETE TASK
# ====================================================================

@task_bp.route('/<task_id>/delete', methods=['POST'])
@login_required
def delete(task_id):
    """Delete a task"""
    from models.task_models import Task
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access - only admins can delete
    if current_user.role != 'Admin':
        flash('Only administrators can delete tasks.', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    try:
        task_title = task.title
        db.session.delete(task)
        db.session.commit()
        flash(f'Task "{task_title}" deleted successfully!', 'success')
        return redirect(url_for('task.task_list'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting task: {str(e)}', 'danger')
        return redirect(url_for('task.view', task_id=task_id))

# ====================================================================
# TASK ACTIONS
# ====================================================================

@task_bp.route('/<task_id>/update-status', methods=['POST'])
@login_required
def update_status(task_id):
    """Update task status"""
    from models.task_models import Task
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access based on role
    if not has_task_access(task, current_user):
        flash('Access denied. You do not have permission to update this task.', 'danger')
        return redirect(url_for('task.task_list'))
    
    status = request.form.get('status')
    if not status:
        flash('Status is required', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    # Prevent non-admins from closing tasks
    if status in ['Resolved', 'Closed'] and current_user.role != 'Admin':
        flash('Only administrators can close tasks.', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    try:
        old_status = task.status
        task.status = status
        
        if status == 'In Progress' and old_status != 'In Progress':
            task.started_at = datetime.utcnow()
        elif status == 'Resolved':
            task.completed_at = datetime.utcnow()
            task.progress = 100
        
        task.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash(f'Task status updated from {old_status} to {status}', 'success')
        return redirect(url_for('task.view', task_id=task.id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating status: {str(e)}', 'danger')
        return redirect(url_for('task.view', task_id=task_id))

@task_bp.route('/<task_id>/update-progress', methods=['POST'])
@login_required
def update_progress(task_id):
    """Update task progress percentage"""
    from models.task_models import Task
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access based on role
    if not has_task_access(task, current_user):
        flash('Access denied. You do not have permission to update this task.', 'danger')
        return redirect(url_for('task.task_list'))
    
    progress = request.form.get('progress', 0)
    try:
        progress = int(progress)
        if progress < 0 or progress > 100:
            flash('Progress must be between 0 and 100', 'danger')
            return redirect(url_for('task.view', task_id=task_id))
    except ValueError:
        flash('Invalid progress value', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    try:
        task.progress = progress
        if progress == 100 and task.status != 'Resolved':
            task.status = 'Resolved'
            task.completed_at = datetime.utcnow()
        elif progress < 100 and task.status == 'Resolved':
            task.status = 'In Progress'
        
        task.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash(f'Task progress updated to {progress}%', 'success')
        return redirect(url_for('task.view', task_id=task.id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating progress: {str(e)}', 'danger')
        return redirect(url_for('task.view', task_id=task_id))

@task_bp.route('/<task_id>/assign', methods=['POST'])
@login_required
def assign(task_id):
    """Assign task to user"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        task = Task.query.get(str(task_id))
    except (ValueError, AttributeError):
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            task = Task.query.filter_by(id=str(task_id)).first()
    
    if not task:
        flash('Task not found', 'danger')
        return redirect(url_for('task.task_list'))
    
    # Check access - only admins and supervisors can assign
    if current_user.role not in ['Admin', 'Supervisor']:
        flash('Only administrators and supervisors can assign tasks.', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    # For supervisors, check department access
    if current_user.role == 'Supervisor' and task.department != current_user.department:
        flash('You can only assign tasks in your department.', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    user_id = request.form.get('user_id')
    if not user_id:
        flash('User is required', 'danger')
        return redirect(url_for('task.view', task_id=task_id))
    
    try:
        user = User.query.get(str(user_id))
        if not user:
            flash('User not found', 'danger')
            return redirect(url_for('task.view', task_id=task_id))
        
        # Validate department for supervisors
        if current_user.role == 'Supervisor' and user.department != current_user.department:
            flash('Can only assign to users in your department', 'danger')
            return redirect(url_for('task.view', task_id=task_id))
        
        old_assigned_to = task.assigned_to
        task.assigned_to = user.id
        task.assigned_by = str(current_user.id)
        task.assigned_at = datetime.utcnow()
        task.status = 'Assigned' if task.status == 'New' else task.status
        task.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        if old_assigned_to:
            flash(f'Task reassigned from previous user to {user.first_name} {user.last_name}', 'success')
        else:
            flash(f'Task assigned to {user.first_name} {user.last_name}', 'success')
        
        return redirect(url_for('task.view', task_id=task.id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error assigning task: {str(e)}', 'danger')
        return redirect(url_for('task.view', task_id=task_id))

# ====================================================================
# BULK OPERATIONS
# ====================================================================

@task_bp.route('/bulk/assign', methods=['POST'])
@login_required
def bulk_assign():
    """Bulk assign tasks"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        task_ids = request.form.get('task_ids', '').split(',')
        user_id = request.form.get('user_id')
        
        if not task_ids or not task_ids[0]:
            flash('Please select tasks to assign.', 'warning')
            return redirect(request.referrer or url_for('task.task_list'))
        
        if not user_id:
            flash('Please select a user to assign tasks to.', 'warning')
            return redirect(request.referrer or url_for('task.task_list'))
        
        user = User.query.get(str(user_id))
        if not user:
            flash('User not found.', 'danger')
            return redirect(request.referrer or url_for('task.task_list'))
        
        # Validate access
        updated_count = 0
        for task_id in task_ids:
            try:
                task = Task.query.get(str(task_id))
                if not task:
                    continue
                
                # Check access based on role
                if not has_task_access(task, current_user):
                    continue
                
                # Validate department for supervisors
                if current_user.role == 'Supervisor' and user.department != current_user.department:
                    continue
                
                task.assigned_to = user.id
                task.assigned_by = str(current_user.id)
                task.assigned_at = datetime.utcnow()
                task.status = 'Assigned' if task.status == 'New' else task.status
                task.updated_at = datetime.utcnow()
                updated_count += 1
                
            except (ValueError, Exception):
                continue
        
        db.session.commit()
        
        flash(f'Assigned {updated_count} task(s) to {user.first_name} {user.last_name}.', 'success')
        return redirect(request.referrer or url_for('task.task_list'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error assigning tasks: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('task.task_list'))

@task_bp.route('/bulk/update-status', methods=['POST'])
@login_required
def bulk_update_status():
    """Bulk update task status"""
    from models.task_models import Task
    
    try:
        task_ids = request.form.get('task_ids', '').split(',')
        status = request.form.get('status')
        
        if not task_ids or not task_ids[0]:
            flash('Please select tasks to update.', 'warning')
            return redirect(request.referrer or url_for('task.task_list'))
        
        if not status:
            flash('Please select a status.', 'warning')
            return redirect(request.referrer or url_for('task.task_list'))
        
        # Prevent non-admins from closing tasks
        if status in ['Resolved', 'Closed'] and current_user.role != 'Admin':
            flash('Only administrators can close tasks.', 'danger')
            return redirect(request.referrer or url_for('task.task_list'))
        
        # Update tasks
        updated_count = 0
        for task_id in task_ids:
            try:
                task = Task.query.get(str(task_id))
                if not task:
                    continue
                
                # Check access based on role
                if not has_task_access(task, current_user):
                    continue
                
                old_status = task.status
                task.status = status
                
                if status == 'In Progress' and old_status != 'In Progress':
                    task.started_at = datetime.utcnow()
                elif status == 'Resolved':
                    task.completed_at = datetime.utcnow()
                    task.progress = 100
                
                task.updated_at = datetime.utcnow()
                updated_count += 1
                
            except (ValueError, Exception):
                continue
        
        db.session.commit()
        
        flash(f'Updated status for {updated_count} task(s) to {status}.', 'success')
        return redirect(request.referrer or url_for('task.task_list'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating task status: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('task.task_list'))

@task_bp.route('/bulk/delete', methods=['POST'])
@login_required
def bulk_delete():
    """Bulk delete tasks"""
    from models.task_models import Task
    
    try:
        task_ids = request.form.get('task_ids', '').split(',')
        
        if not task_ids or not task_ids[0]:
            flash('Please select tasks to delete.', 'warning')
            return redirect(request.referrer or url_for('task.task_list'))
        
        # Only admins can bulk delete
        if current_user.role != 'Admin':
            flash('Only administrators can delete tasks.', 'danger')
            return redirect(request.referrer or url_for('task.task_list'))
        
        # Delete tasks
        deleted_count = 0
        for task_id in task_ids:
            try:
                task = Task.query.get(str(task_id))
                if not task:
                    continue
                
                db.session.delete(task)
                deleted_count += 1
                
            except (ValueError, Exception):
                continue
        
        db.session.commit()
        
        flash(f'Deleted {deleted_count} task(s).', 'success')
        return redirect(request.referrer or url_for('task.task_list'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting tasks: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('task.task_list'))

# ====================================================================
# API ENDPOINTS
# ====================================================================

@task_bp.route('/api/stats', methods=['GET'])
@login_required
def task_stats():
    """Get task statistics"""
    from models.task_models import Task
    from models.user_models import User
    
    stats = {}
    user_id_str = str(current_user.id)
    
    if current_user.role == 'Admin':
        stats['total_tasks'] = Task.query.count()
        stats['my_tasks'] = Task.query.filter(Task.assigned_to == user_id_str).count()
        stats['open_tasks'] = Task.query.filter(Task.status.in_(['New', 'Assigned', 'In Progress'])).count()
        # FIXED: Use offset-naive datetime for comparison
        stats['overdue_tasks'] = Task.query.filter(
            Task.due_date < datetime.utcnow().replace(tzinfo=None),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        stats['completed_tasks'] = Task.query.filter_by(status='Resolved').count()
        
        # Department breakdown
        dept_stats = db.session.query(
            Task.department,
            db.func.count(Task.id).label('count')
        ).group_by(Task.department).all()
        stats['department_stats'] = {dept: count for dept, count in dept_stats if dept}
        
        # Status breakdown
        status_stats = db.session.query(
            Task.status,
            db.func.count(Task.id).label('count')
        ).group_by(Task.status).all()
        stats['status_stats'] = {status: count for status, count in status_stats}
        
        # Category breakdown
        category_stats = db.session.query(
            Task.category,
            db.func.count(Task.id).label('count')
        ).group_by(Task.category).all()
        stats['category_stats'] = {category: count for category, count in category_stats if category}
        
    elif current_user.role == 'Supervisor':
        stats['total_tasks'] = Task.query.filter_by(department=current_user.department).count()
        stats['my_tasks'] = Task.query.filter(Task.assigned_to == user_id_str).count()
        stats['open_tasks'] = Task.query.filter(
            Task.department == current_user.department,
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        # FIXED: Use offset-naive datetime for comparison
        stats['overdue_tasks'] = Task.query.filter(
            Task.department == current_user.department,
            Task.due_date < datetime.utcnow().replace(tzinfo=None),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        stats['completed_tasks'] = Task.query.filter_by(
            department=current_user.department,
            status='Resolved'
        ).count()
        
    else:
        stats['total_tasks'] = Task.query.filter(Task.assigned_to == user_id_str).count()
        stats['my_tasks'] = stats['total_tasks']
        stats['open_tasks'] = Task.query.filter(
            Task.assigned_to == user_id_str,
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        # FIXED: Use offset-naive datetime for comparison
        stats['overdue_tasks'] = Task.query.filter(
            Task.assigned_to == user_id_str,
            Task.due_date < datetime.utcnow().replace(tzinfo=None),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        stats['completed_tasks'] = Task.query.filter_by(
            assigned_to=user_id_str,
            status='Resolved'
        ).count()
    
    return jsonify(stats)

@task_bp.route('/api/tasks', methods=['GET'])
@login_required
def api_tasks():
    """API endpoint to get tasks as JSON"""
    from models.task_models import Task
    from models.user_models import User
    
    user_id_str = str(current_user.id)
    
    # Similar filtering logic as task_list
    if current_user.role == 'Admin':
        query = Task.query
    elif current_user.role == 'Supervisor':
        query = Task.query.filter_by(department=current_user.department)
    else:
        query = Task.query.filter(Task.assigned_to == user_id_str)
    
    # Apply filters
    status = request.args.get('status')
    if status and status != 'all':
        query = query.filter_by(status=status)
    
    priority = request.args.get('priority')
    if priority and priority != 'all':
        query = query.filter_by(priority=priority)
    
    department = request.args.get('department')
    if department and department != 'all' and current_user.role == 'Admin':
        query = query.filter_by(department=department)
    
    # Get tasks
    tasks = query.order_by(Task.created_at.desc()).limit(100).all()
    
    # Serialize tasks
    tasks_data = []
    for task in tasks:
        assigned_user = None
        if task.assigned_to:
            user = User.query.get(str(task.assigned_to))
            if user:
                assigned_user = {
                    'id': user.id,
                    'name': f"{user.first_name} {user.last_name}",
                    'email': user.email
                }
        
        # FIXED: Use offset-naive datetime for comparison
        now = datetime.utcnow().replace(tzinfo=None)
        task_due_date = task.due_date.replace(tzinfo=None) if task.due_date and task.due_date.tzinfo is not None else task.due_date
        
        tasks_data.append({
            'id': task.id,
            'title': task.title,
            'description': task.description,
            'status': task.status,
            'priority': task.priority,
            'department': task.department,
            'category': task.category,
            'due_date': task.due_date.isoformat() if task.due_date else None,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'assigned_to': assigned_user,
            'progress': task.progress,
            'is_overdue': task_due_date and task_due_date < now and task.status in ['New', 'Assigned', 'In Progress']
        })
    
    return jsonify({'tasks': tasks_data})

# ====================================================================
# EXPORT FUNCTIONS
# ====================================================================

@task_bp.route('/export/csv', methods=['GET'])
@login_required
def export_csv():
    """Export tasks to CSV"""
    from models.task_models import Task
    from models.user_models import User
    
    user_id_str = str(current_user.id)
    
    # Similar filtering logic as task_list
    if current_user.role == 'Admin':
        query = Task.query
    elif current_user.role == 'Supervisor':
        query = Task.query.filter_by(department=current_user.department)
    else:
        query = Task.query.filter(Task.assigned_to == user_id_str)
    
    tasks = query.order_by(Task.created_at.desc()).all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Title', 'Description', 'Status', 'Priority', 'Department', 
                    'Category', 'Assigned To', 'Due Date', 'Created At', 'Progress'])
    
    # Write data
    for task in tasks:
        assigned_user = 'Unassigned'
        if task.assigned_to:
            user = User.query.get(str(task.assigned_to))
            if user:
                assigned_user = f"{user.first_name} {user.last_name}"
        
        writer.writerow([
            task.id,
            task.title,
            task.description or '',
            task.status,
            task.priority,
            task.department or '',
            task.category or '',
            assigned_user,
            task.due_date.strftime('%Y-%m-%d') if task.due_date else '',
            task.created_at.strftime('%Y-%m-%d %H:%M') if task.created_at else '',
            task.progress or 0
        ])
    
    # Return CSV file
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'tasks_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
    )

# ====================================================================
# ERROR HANDLERS FOR BLUEPRINT
# ====================================================================

@task_bp.errorhandler(404)
def task_not_found(error):
    flash('Task not found', 'danger')
    return redirect(url_for('task.task_list'))

@task_bp.errorhandler(403)
def access_denied(error):
    flash('Access denied. You do not have permission to perform this action.', 'danger')
    return redirect(url_for('task.task_list'))

@task_bp.errorhandler(500)
def server_error(error):
    flash('An internal server error occurred. Please try again later.', 'danger')
    return redirect(url_for('task.task_list'))

# ====================================================================
# ALIAS ENDPOINTS FOR TEMPLATE COMPATIBILITY
# ====================================================================

@task_bp.route('/<task_id>/detail')
@login_required
def task_detail(task_id):
    """Alias for task.view to fix template compatibility"""
    return view(task_id)

@task_bp.route('/<task_id>/update-task', methods=['POST'])
@login_required
def update_task(task_id):
    """Alias for edit with POST method"""
    return edit(task_id)

@task_bp.route('/stats')
@login_required
def task_stats_redirect():
    """Redirect to task stats API"""
    return redirect(url_for('task.task_stats'))