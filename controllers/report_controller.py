from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for, send_file
from flask_login import login_required, current_user
from datetime import datetime, timedelta
from functools import wraps
import json
import csv
import io

report_bp = Blueprint('report', __name__, url_prefix='/reports')

# Helper decorator for role-based access
def require_roles(*roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if current_user.role not in roles:
                flash('Access denied. Insufficient permissions.', 'danger')
                return redirect(url_for('report.dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@report_bp.route('/')
@login_required
@require_roles('Admin', 'Supervisor')
def dashboard():
    """Reports dashboard"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        total_tasks = Task.query.count()
        completed_tasks = Task.query.filter_by(status='Resolved').count()
        
        # Calculate overdue tasks
        overdue_tasks = Task.query.filter(
            Task.due_date < datetime.utcnow(),
            Task.status.in_(['New', 'Assigned', 'In Progress'])
        ).count()
        
        total_users = User.query.filter_by(is_active=True).count()
        
        # Calculate SLA compliance
        sla_met = Task.query.filter(
            Task.status == 'Resolved',
            Task.completed_at <= Task.due_date
        ).count()
        
        sla_total = Task.query.filter(
            Task.status == 'Resolved',
            Task.due_date.isnot(None)
        ).count()
        
        sla_compliance = (sla_met / sla_total * 100) if sla_total > 0 else 0
        
        # Calculate completion rate
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Get recent activity
        from models.task_models import TaskActivity
        recent_activity = TaskActivity.query.order_by(
            TaskActivity.created_at.desc()
        ).limit(10).all()
        
        # Get category distribution
        categories = {}
        category_tasks = Task.query.with_entities(Task.category, Task.status).all()
        for category, status in category_tasks:
            if category:
                if category not in categories:
                    categories[category] = {'total': 0, 'completed': 0}
                categories[category]['total'] += 1
                if status == 'Resolved':
                    categories[category]['completed'] += 1
        
        return render_template('reports/dashboard.html',
                             total_tasks=total_tasks,
                             completed_tasks=completed_tasks,
                             overdue_tasks=overdue_tasks,
                             total_users=total_users,
                             sla_compliance=round(sla_compliance, 1),
                             completion_rate=round(completion_rate, 1),
                             recent_activity=recent_activity,
                             categories=categories)
    
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('reports/dashboard.html',
                             total_tasks=0,
                             completed_tasks=0,
                             overdue_tasks=0,
                             total_users=0,
                             sla_compliance=0,
                             completion_rate=0,
                             recent_activity=[],
                             categories={})

@report_bp.route('/sla')
@login_required
@require_roles('Admin', 'Supervisor')
def sla():
    """SLA reports"""
    from models.task_models import Task
    
    try:
        # Calculate SLA metrics
        tasks = Task.query.all()
        
        sla_data = {
            'met': 0,
            'missed': 0,
            'pending': 0,
            'no_due_date': 0
        }
        
        for task in tasks:
            if not task.due_date:
                sla_data['no_due_date'] += 1
            elif task.status == 'Resolved':
                if task.completed_at and task.due_date and task.completed_at <= task.due_date:
                    sla_data['met'] += 1
                else:
                    sla_data['missed'] += 1
            else:
                sla_data['pending'] += 1
        
        # Get trends for last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_tasks = Task.query.filter(
            Task.created_at >= thirty_days_ago
        ).all()
        
        # Calculate daily SLA compliance
        daily_compliance = {}
        for task in recent_tasks:
            if task.status == 'Resolved' and task.completed_at:
                date_str = task.completed_at.strftime('%Y-%m-%d')
                if date_str not in daily_compliance:
                    daily_compliance[date_str] = {'met': 0, 'total': 0}
                
                daily_compliance[date_str]['total'] += 1
                if task.due_date and task.completed_at <= task.due_date:
                    daily_compliance[date_str]['met'] += 1
        
        # Convert to list for chart
        trend_data = []
        for date, data in sorted(daily_compliance.items()):
            compliance_rate = (data['met'] / data['total'] * 100) if data['total'] > 0 else 0
            trend_data.append({
                'date': date,
                'compliance_rate': round(compliance_rate, 1),
                'total': data['total'],
                'met': data['met']
            })
        
        return render_template('reports/sla.html', 
                             sla_data=sla_data,
                             trend_data=trend_data)
    
    except Exception as e:
        flash(f'Error loading SLA report: {str(e)}', 'danger')
        return render_template('reports/sla.html',
                             sla_data={'met': 0, 'missed': 0, 'pending': 0, 'no_due_date': 0},
                             trend_data=[])

@report_bp.route('/performance')
@login_required
@require_roles('Admin', 'Supervisor')
def performance():
    """Performance reports"""
    from models.task_models import Task
    from models.user_models import User
    
    try:
        # Get all active users
        users = User.query.filter_by(is_active=True).all()
        performance_data = []
        
        for user in users:
            # Get user's tasks
            tasks = Task.query.filter_by(assigned_to=user.id).all()
            completed = [t for t in tasks if t.status == 'Resolved']
            in_progress = [t for t in tasks if t.status == 'In Progress']
            overdue = [t for t in tasks if t.due_date and t.due_date < datetime.utcnow() and t.status != 'Resolved']
            
            # Calculate metrics
            total = len(tasks)
            completion_rate = (len(completed) / total * 100) if total > 0 else 0
            
            # Calculate average completion time
            avg_completion_days = 0
            if completed:
                total_days = 0
                for task in completed:
                    if task.completed_at and task.created_at:
                        days = (task.completed_at - task.created_at).days
                        if days >= 0:
                            total_days += days
                avg_completion_days = total_days / len(completed) if len(completed) > 0 else 0
            
            # Calculate SLA compliance for this user
            user_sla_met = 0
            user_sla_total = 0
            for task in completed:
                if task.due_date:
                    user_sla_total += 1
                    if task.completed_at and task.completed_at <= task.due_date:
                        user_sla_met += 1
            
            sla_compliance = (user_sla_met / user_sla_total * 100) if user_sla_total > 0 else 0
            
            performance_data.append({
                'user': user,
                'total_tasks': total,
                'completed': len(completed),
                'in_progress': len(in_progress),
                'overdue': len(overdue),
                'completion_rate': round(completion_rate, 1),
                'avg_completion_days': round(avg_completion_days, 1),
                'sla_compliance': round(sla_compliance, 1)
            })
        
        # Sort by completion rate (highest first)
        performance_data.sort(key=lambda x: x['completion_rate'], reverse=True)
        
        # Get department performance
        department_performance = {}
        for data in performance_data:
            dept = data['user'].department or 'Unassigned'
            if dept not in department_performance:
                department_performance[dept] = {
                    'total_users': 0,
                    'total_tasks': 0,
                    'total_completed': 0,
                    'avg_completion_rate': 0
                }
            
            department_performance[dept]['total_users'] += 1
            department_performance[dept]['total_tasks'] += data['total_tasks']
            department_performance[dept]['total_completed'] += data['completed']
        
        # Calculate department averages
        for dept in department_performance:
            dept_data = department_performance[dept]
            if dept_data['total_tasks'] > 0:
                dept_data['avg_completion_rate'] = round(
                    (dept_data['total_completed'] / dept_data['total_tasks'] * 100), 1
                )
        
        return render_template('reports/performance.html', 
                             performance_data=performance_data,
                             department_performance=department_performance)
    
    except Exception as e:
        flash(f'Error loading performance report: {str(e)}', 'danger')
        return render_template('reports/performance.html',
                             performance_data=[],
                             department_performance={})

@report_bp.route('/generate', methods=['GET', 'POST'])
@login_required
@require_roles('Admin', 'Supervisor')
def generate_report():
    """Generate custom reports"""
    if request.method == 'POST':
        try:
            # Get form data
            report_type = request.form.get('report_type')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            format_type = request.form.get('format', 'html')
            
            # Validate dates
            start = None
            end = None
            if start_date:
                start = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate report based on type
            if report_type == 'sla':
                if format_type == 'csv':
                    return export_sla_csv(start, end)
                else:
                    flash('SLA report generated successfully!', 'success')
                    return redirect(url_for('report.sla'))
            
            elif report_type == 'performance':
                if format_type == 'csv':
                    return export_performance_csv(start, end)
                else:
                    flash('Performance report generated successfully!', 'success')
                    return redirect(url_for('report.performance'))
            
            elif report_type == 'summary':
                if format_type == 'csv':
                    return export_summary_csv(start, end)
                else:
                    flash('Task summary report generated successfully!', 'success')
                    return redirect(url_for('report.dashboard'))
            
            else:
                flash('Please select a valid report type', 'danger')
                return redirect(url_for('report.generate_report'))
                
        except Exception as e:
            flash(f'Error generating report: {str(e)}', 'danger')
            return redirect(url_for('report.generate_report'))
    
    # GET request - show the form
    return render_template('reports/generate.html')

@report_bp.route('/export/<report_type>')
@login_required
@require_roles('Admin', 'Supervisor')
def export_report(report_type):
    """Export reports in various formats"""
    if report_type == 'tasks':
        return export_tasks_csv()
    elif report_type == 'sla':
        return export_sla_csv()
    elif report_type == 'performance':
        return export_performance_csv()
    elif report_type == 'summary':
        return export_summary_csv()
    else:
        return "Report type not found", 404

@report_bp.route('/api/stats')
@login_required
def report_stats():
    """Get report statistics (API endpoint)"""
    from models.task_models import Task
    
    try:
        # Get timeframe from request (default: last 30 days)
        days = int(request.args.get('days', 30))
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Task creation trend
        tasks = Task.query.filter(Task.created_at >= start_date).all()
        
        # Group by date
        daily_counts = {}
        for task in tasks:
            date_str = task.created_at.strftime('%Y-%m-%d')
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        # Convert to sorted list
        trend_data = [{'date': date, 'count': count} 
                     for date, count in sorted(daily_counts.items())]
        
        # Completion rate
        total_tasks = Task.query.count()
        completed_tasks = Task.query.filter_by(status='Resolved').count()
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # SLA compliance
        sla_met = Task.query.filter(
            Task.status == 'Resolved',
            Task.completed_at <= Task.due_date
        ).count()
        
        sla_total = Task.query.filter(
            Task.status == 'Resolved',
            Task.due_date.isnot(None)
        ).count()
        
        sla_compliance = (sla_met / sla_total * 100) if sla_total > 0 else 0
        
        return jsonify({
            'success': True,
            'trend_data': trend_data,
            'completion_rate': round(completion_rate, 1),
            'sla_compliance': round(sla_compliance, 1),
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@report_bp.route('/api/sla-details')
@login_required
def sla_details():
    """Get detailed SLA breakdown by department"""
    from models.task_models import Task
    from models.database import db
    from sqlalchemy import func
    
    try:
        # Query tasks grouped by department
        results = db.session.query(
            Task.department,
            func.count(Task.id).label('total_tasks'),
            func.sum(func.case((Task.status == 'Resolved', 1), else_=0)).label('completed'),
            func.sum(func.case((Task.status.in_(['New', 'Assigned', 'In Progress']), 1), else_=0)).label('pending'),
            func.sum(func.case((
                (Task.status == 'Resolved') & 
                (Task.completed_at <= Task.due_date), 1), else_=0)).label('sla_met'),
            func.sum(func.case((
                (Task.status == 'Resolved') & 
                (Task.completed_at > Task.due_date), 1), else_=0)).label('sla_missed')
        ).group_by(Task.department).all()
        
        departments = []
        for dept, total, completed, pending, sla_met, sla_missed in results:
            departments.append({
                'department': dept or 'Uncategorized',
                'total_tasks': total or 0,
                'completed': completed or 0,
                'pending': pending or 0,
                'sla_met': sla_met or 0,
                'sla_missed': sla_missed or 0,
                'total_with_due_date': (sla_met or 0) + (sla_missed or 0) + (pending or 0)
            })
        
        return jsonify({
            'success': True,
            'departments': departments
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@report_bp.route('/api/performance-details')
@login_required
def performance_details():
    """Get detailed performance data"""
    from models.user_models import User
    from models.task_models import Task
    
    try:
        users = User.query.filter_by(is_active=True).all()
        user_data = []
        
        for user in users:
            tasks = Task.query.filter_by(assigned_to=user.id).all()
            completed = [t for t in tasks if t.status == 'Resolved']
            
            # Calculate metrics
            total = len(tasks)
            completion_rate = (len(completed) / total * 100) if total > 0 else 0
            
            user_data.append({
                'user_id': user.id,
                'user_name': user.full_name,
                'department': user.department or 'Unassigned',
                'total_tasks': total,
                'completed': len(completed),
                'completion_rate': round(completion_rate, 1)
            })
        
        return jsonify({
            'success': True,
            'users': user_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Helper functions for CSV exports
def export_tasks_csv(start_date=None, end_date=None):
    """Export tasks to CSV"""
    from models.task_models import Task
    from models.user_models import User
    
    # Build query
    query = Task.query
    
    if start_date:
        query = query.filter(Task.created_at >= start_date)
    
    if end_date:
        query = query.filter(Task.created_at <= end_date)
    
    tasks = query.all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Task ID', 'Title', 'Status', 'Priority', 'Category',
        'Department', 'Assigned To', 'Email', 'Created At', 
        'Due Date', 'Completed At', 'Resolution Summary'
    ])
    
    # Write data
    for task in tasks:
        assigned_user = User.query.get(task.assigned_to) if task.assigned_to else None
        
        writer.writerow([
            task.id,
            task.title,
            task.status,
            task.priority,
            task.category,
            task.department,
            assigned_user.full_name if assigned_user else 'Unassigned',
            assigned_user.email if assigned_user else '',
            task.created_at.strftime('%Y-%m-%d %H:%M') if task.created_at else '',
            task.due_date.strftime('%Y-%m-%d') if task.due_date else '',
            task.completed_at.strftime('%Y-%m-%d %H:%M') if task.completed_at else '',
            task.resolution_summary or ''
        ])
    
    output.seek(0)
    filename = f'tasks_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def export_sla_csv(start_date=None, end_date=None):
    """Export SLA report to CSV"""
    from models.task_models import Task
    
    # Build query
    query = Task.query
    
    if start_date:
        query = query.filter(Task.created_at >= start_date)
    
    if end_date:
        query = query.filter(Task.created_at <= end_date)
    
    tasks = query.all()
    
    # Calculate SLA metrics
    sla_data = {
        'met': 0,
        'missed': 0,
        'pending': 0,
        'no_due_date': 0
    }
    
    for task in tasks:
        if not task.due_date:
            sla_data['no_due_date'] += 1
        elif task.status == 'Resolved':
            if task.completed_at and task.due_date and task.completed_at <= task.due_date:
                sla_data['met'] += 1
            else:
                sla_data['missed'] += 1
        else:
            sla_data['pending'] += 1
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Metric', 'Count', 'Percentage'])
    
    total_with_due_date = sla_data['met'] + sla_data['missed'] + sla_data['pending']
    total_all = sum(sla_data.values())
    
    # Write data
    writer.writerow(['SLA Met', sla_data['met'], 
                    f"{(sla_data['met'] / total_with_due_date * 100):.1f}%" if total_with_due_date > 0 else '0%'])
    writer.writerow(['SLA Missed', sla_data['missed'], 
                    f"{(sla_data['missed'] / total_with_due_date * 100):.1f}%" if total_with_due_date > 0 else '0%'])
    writer.writerow(['Pending', sla_data['pending'], 
                    f"{(sla_data['pending'] / total_with_due_date * 100):.1f}%" if total_with_due_date > 0 else '0%'])
    writer.writerow(['No Due Date', sla_data['no_due_date'], 
                    f"{(sla_data['no_due_date'] / total_all * 100):.1f}%" if total_all > 0 else '0%'])
    writer.writerow([])  # Empty row
    writer.writerow(['Total with Due Date', total_with_due_date, ''])
    writer.writerow(['Total All Tasks', total_all, ''])
    
    if total_with_due_date > 0:
        compliance_rate = (sla_data['met'] / total_with_due_date * 100)
        writer.writerow(['SLA Compliance Rate', f'{compliance_rate:.1f}%', ''])
    
    output.seek(0)
    filename = f'sla_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def export_performance_csv(start_date=None, end_date=None):
    """Export performance report to CSV"""
    from models.user_models import User
    from models.task_models import Task
    
    users = User.query.filter_by(is_active=True).all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'User', 'Department', 'Email', 'Total Tasks', 'Completed',
        'In Progress', 'Overdue', 'Completion Rate %', 'Avg Completion Days',
        'SLA Compliance %'
    ])
    
    # Write data
    for user in users:
        # Get user's tasks with date filter
        query = Task.query.filter_by(assigned_to=user.id)
        
        if start_date:
            query = query.filter(Task.created_at >= start_date)
        
        if end_date:
            query = query.filter(Task.created_at <= end_date)
        
        tasks = query.all()
        
        completed = [t for t in tasks if t.status == 'Resolved']
        in_progress = [t for t in tasks if t.status == 'In Progress']
        overdue = [t for t in tasks if t.due_date and t.due_date < datetime.utcnow() and t.status != 'Resolved']
        
        # Calculate metrics
        total = len(tasks)
        completion_rate = (len(completed) / total * 100) if total > 0 else 0
        
        # Calculate average completion time
        avg_completion_days = 0
        if completed:
            total_days = 0
            for task in completed:
                if task.completed_at and task.created_at:
                    days = (task.completed_at - task.created_at).days
                    if days >= 0:
                        total_days += days
            avg_completion_days = total_days / len(completed) if len(completed) > 0 else 0
        
        # Calculate SLA compliance
        user_sla_met = 0
        user_sla_total = 0
        for task in completed:
            if task.due_date:
                user_sla_total += 1
                if task.completed_at and task.completed_at <= task.due_date:
                    user_sla_met += 1
        
        sla_compliance = (user_sla_met / user_sla_total * 100) if user_sla_total > 0 else 0
        
        writer.writerow([
            user.full_name,
            user.department or 'Unassigned',
            user.email,
            total,
            len(completed),
            len(in_progress),
            len(overdue),
            f'{completion_rate:.1f}',
            f'{avg_completion_days:.1f}',
            f'{sla_compliance:.1f}'
        ])
    
    output.seek(0)
    filename = f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def export_summary_csv(start_date=None, end_date=None):
    """Export task summary report to CSV"""
    from models.task_models import Task
    
    # Build query
    query = Task.query
    
    if start_date:
        query = query.filter(Task.created_at >= start_date)
    
    if end_date:
        query = query.filter(Task.created_at <= end_date)
    
    tasks = query.all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write summary header
    writer.writerow(['Task Management System - Summary Report'])
    writer.writerow([f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
    if start_date:
        writer.writerow([f'Start Date: {start_date.strftime("%Y-%m-%d")}'])
    if end_date:
        writer.writerow([f'End Date: {end_date.strftime("%Y-%m-%d")}'])
    writer.writerow([])
    
    # Summary statistics
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.status == 'Resolved'])
    pending_tasks = len([t for t in tasks if t.status in ['New', 'Assigned', 'In Progress']])
    overdue_tasks = len([t for t in tasks if t.due_date and t.due_date < datetime.utcnow() and t.status != 'Resolved'])
    
    writer.writerow(['Summary Statistics'])
    writer.writerow(['Total Tasks', total_tasks])
    writer.writerow(['Completed Tasks', completed_tasks])
    writer.writerow(['Pending Tasks', pending_tasks])
    writer.writerow(['Overdue Tasks', overdue_tasks])
    writer.writerow(['Completion Rate', f'{(completed_tasks / total_tasks * 100):.1f}%' if total_tasks > 0 else '0%'])
    writer.writerow([])
    
    # Status breakdown
    writer.writerow(['Status Breakdown'])
    status_counts = {}
    for task in tasks:
        status_counts[task.status] = status_counts.get(task.status, 0) + 1
    
    for status, count in status_counts.items():
        writer.writerow([status, count, f'{(count / total_tasks * 100):.1f}%' if total_tasks > 0 else '0%'])
    writer.writerow([])
    
    # Priority breakdown
    writer.writerow(['Priority Breakdown'])
    priority_counts = {}
    for task in tasks:
        priority_counts[task.priority] = priority_counts.get(task.priority, 0) + 1
    
    for priority, count in priority_counts.items():
        writer.writerow([priority, count, f'{(count / total_tasks * 100):.1f}%' if total_tasks > 0 else '0%'])
    writer.writerow([])
    
    # Category breakdown
    writer.writerow(['Category Breakdown'])
    category_counts = {}
    for task in tasks:
        if task.category:
            category_counts[task.category] = category_counts.get(task.category, 0) + 1
    
    for category, count in category_counts.items():
        writer.writerow([category, count, f'{(count / total_tasks * 100):.1f}%' if total_tasks > 0 else '0%'])
    
    output.seek(0)
    filename = f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

# Error handler for report blueprint
@report_bp.errorhandler(404)
def report_not_found(error):
    flash('Report not found', 'danger')
    return redirect(url_for('report.dashboard'))