"""
Admin Controller - Administrative functions and system management
"""

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for, send_file
from flask_login import login_required, current_user
from models.database import db
from models.user_models import User
from models.task_models import Task, WorkflowTemplate
from datetime import datetime, timedelta
import json
import os
import csv
from io import StringIO

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
@login_required
def admin_dashboard():
    """Admin dashboard view"""
    if current_user.role != 'Admin':
        flash('Access denied. Admin rights required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # System statistics
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_tasks = Task.query.count()
    completed_tasks = Task.query.filter_by(status='Closed').count()
    
    # Recent activity
    recent_tasks = Task.query.order_by(Task.created_at.desc()).limit(10).all()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    
    # System health
    db_size = os.path.getsize('instance/database.db') / (1024 * 1024)  # MB
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         active_users=active_users,
                         total_tasks=total_tasks,
                         completed_tasks=completed_tasks,
                         recent_tasks=recent_tasks,
                         recent_users=recent_users,
                         db_size=db_size)

@admin_bp.route('/system-settings', methods=['GET', 'POST'])
@login_required
def system_settings():
    """System settings management"""
    if current_user.role != 'Admin':
        flash('Access denied. Admin rights required.', 'danger')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Update system settings
        settings = {
            'sla_critical': int(request.form.get('sla_critical', 2)),
            'sla_high': int(request.form.get('sla_high', 4)),
            'sla_medium': int(request.form.get('sla_medium', 24)),
            'sla_low': int(request.form.get('sla_low', 72)),
            'auto_logout_minutes': int(request.form.get('auto_logout', 120)),
            'max_login_attempts': int(request.form.get('max_login_attempts', 5)),
            'task_retention_days': int(request.form.get('task_retention', 365)),
            'log_retention_days': int(request.form.get('log_retention', 90)),
            'email_notifications': request.form.get('email_notifications') == 'on',
            'auto_assign_tasks': request.form.get('auto_assign_tasks') == 'on',
            'enable_ml_predictions': request.form.get('enable_ml_predictions') == 'on'
        }
        
        # Save settings to file or database
        with open('instance/system_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        
        flash('System settings updated successfully!', 'success')
        return redirect(url_for('admin.system_settings'))
    
    # Load existing settings
    try:
        with open('instance/system_settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        settings = {}
    
    return render_template('admin/system_settings.html', settings=settings)

@admin_bp.route('/backup', methods=['POST'])
@login_required
def create_backup():
    """Create system backup"""
    if current_user.role != 'Admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        import shutil
        from datetime import datetime
        
        # Create backup directory if not exists
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f'{backup_dir}/backup_{timestamp}.db'
        
        # Copy database file
        shutil.copy2('instance/database.db', backup_file)
        
        # Also backup settings
        settings_file = f'{backup_dir}/settings_{timestamp}.json'
        if os.path.exists('instance/system_settings.json'):
            shutil.copy2('instance/system_settings.json', settings_file)
        
        return jsonify({
            'success': True,
            'message': 'Backup created successfully',
            'file': backup_file
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/backup/list')
@login_required
def list_backups():
    """List available backups"""
    if current_user.role != 'Admin':
        return jsonify({'error': 'Access denied'}), 403
    
    backups = []
    backup_dir = 'backups'
    
    if os.path.exists(backup_dir):
        for file in os.listdir(backup_dir):
            file_path = os.path.join(backup_dir, file)
            if os.path.isfile(file_path):
                backups.append({
                    'name': file,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
    
    return jsonify({'backups': backups})

@admin_bp.route('/restore/<backup_name>', methods=['POST'])
@login_required
def restore_backup(backup_name):
    """Restore from backup"""
    if current_user.role != 'Admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        import shutil
        
        backup_path = f'backups/{backup_name}'
        
        if not os.path.exists(backup_path):
            return jsonify({'error': 'Backup not found'}), 404
        
        # Create restore backup first
        restore_backup_dir = 'restore_backups'
        os.makedirs(restore_backup_dir, exist_ok=True)
        restore_file = f'{restore_backup_dir}/before_restore_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        shutil.copy2('instance/database.db', restore_file)
        
        # Restore from backup
        shutil.copy2(backup_path, 'instance/database.db')
        
        return jsonify({
            'success': True,
            'message': 'Backup restored successfully',
            'restore_backup': restore_file
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/audit-log')
@login_required
def audit_log():
    """View system audit log"""
    if current_user.role != 'Admin':
        flash('Access denied. Admin rights required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get filter parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    user_id = request.args.get('user_id')
    action = request.args.get('action')
    
    # Query audit logs (this would come from an AuditLog model)
    # For now, we'll use task activities as audit logs
    query = db.session.query(Task)
    
    if start_date:
        query = query.filter(Task.created_at >= datetime.strptime(start_date, '%Y-%m-%d'))
    if end_date:
        query = query.filter(Task.created_at <= datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))
    if user_id:
        query = query.filter(Task.created_by == user_id)
    
    audit_logs = query.order_by(Task.created_at.desc()).limit(100).all()
    
    users = User.query.all()
    
    return render_template('admin/audit_log.html',
                         audit_logs=audit_logs,
                         users=users)

@admin_bp.route('/export-data')
@login_required
def export_data():
    """Export system data"""
    if current_user.role != 'Admin':
        flash('Access denied. Admin rights required.', 'danger')
        return redirect(url_for('dashboard'))
    
    data_type = request.args.get('type', 'tasks')
    
    if data_type == 'tasks':
        tasks = Task.query.all()
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Task ID', 'Title', 'Category', 'Priority', 'Status',
                        'Created Date', 'Created By', 'Assigned To',
                        'Department', 'SLA Due', 'Resolution Time'])
        
        # Write data
        for task in tasks:
            writer.writerow([
                task.task_id,
                task.title,
                task.category,
                task.priority,
                task.status,
                task.created_at.strftime('%Y-%m-%d %H:%M') if task.created_at else '',
                task.requester.full_name if task.requester else '',
                task.technician.full_name if task.technician else '',
                task.department,
                task.sla_due_date.strftime('%Y-%m-%d %H:%M') if task.sla_due_date else '',
                task.actual_hours
            ])
        
        output.seek(0)
        return send_file(
            StringIO(output.getvalue()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'tasks_export_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    elif data_type == 'users':
        users = User.query.all()
        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['Employee ID', 'Name', 'Email', 'Role', 'Department',
                        'Job Title', 'Status', 'Last Login', 'Tasks Assigned'])
        
        for user in users:
            tasks_assigned = Task.query.filter_by(assigned_to=user.id).count()
            writer.writerow([
                user.employee_id,
                user.full_name,
                user.email,
                user.role,
                user.department,
                user.job_title or '',
                'Active' if user.is_active else 'Inactive',
                user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else '',
                tasks_assigned
            ])
        
        output.seek(0)
        return send_file(
            StringIO(output.getvalue()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'users_export_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    return jsonify({'error': 'Invalid export type'}), 400

@admin_bp.route('/clear-cache', methods=['POST'])
@login_required
def clear_cache():
    """Clear system cache"""
    if current_user.role != 'Admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Clear cache directory
        cache_dir = 'instance/cache'
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
        
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/system-health')
@login_required
def system_health():
    """Check system health"""
    if current_user.role != 'Admin':
        return jsonify({'error': 'Access denied'}), 403
    
    health = {
        'database': {
            'status': 'ok',
            'size_mb': round(os.path.getsize('instance/database.db') / (1024 * 1024), 2)
        },
        'disk_space': {
            'status': 'ok',
            'free_gb': round(shutil.disk_usage('.').free / (1024**3), 2)
        },
        'memory': {
            'status': 'ok',
            'usage_percent': psutil.virtual_memory().percent
        },
        'uptime': {
            'days': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).days
        }
    }
    
    return jsonify(health)