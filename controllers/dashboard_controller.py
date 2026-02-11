"""
Dashboard Controller - Main dashboard and statistics
"""

from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from models.database import db
from models.task_models import Task
from models.user_models import User
from datetime import datetime, timedelta
import json

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/api/stats')
@login_required
def get_stats():
    """Get dashboard statistics via API"""
    
    # Get time range
    days = int(request.args.get('days', 30))
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Base queries
    total_tasks = Task.query.count()
    open_tasks = Task.query.filter(Task.status.in_(['New', 'Acknowledged', 'Assigned', 'In Progress'])).count()
    
    # Overdue tasks
    overdue_tasks = Task.query.filter(
        Task.sla_due_date < datetime.utcnow(),
        Task.status.in_(['New', 'Acknowledged', 'Assigned', 'In Progress'])
    ).count()
    
    # User-specific stats
    if current_user.role in ['Admin', 'Supervisor']:
        my_tasks = Task.query.filter(Task.assigned_to == current_user.id).count()
        team_tasks = Task.query.filter(
            Task.status.in_(['New', 'Acknowledged', 'Assigned', 'In Progress'])
        ).count()
    else:
        my_tasks = Task.query.filter(Task.assigned_to == current_user.id).count()
        team_tasks = 0
    
    # SLA compliance
    completed_tasks = Task.query.filter(
        Task.status == 'Closed',
        Task.created_at >= start_date
    ).count()
    
    sla_compliant_tasks = Task.query.filter(
        Task.status == 'Closed',
        Task.sla_due_date.isnot(None),
        Task.end_time <= Task.sla_due_date,
        Task.created_at >= start_date
    ).count()
    
    sla_compliance = (sla_compliant_tasks / completed_tasks * 100) if completed_tasks > 0 else 100
    
    # Recent tasks
    recent_tasks = Task.query.order_by(Task.created_at.desc()).limit(5).all()
    recent_tasks_data = []
    for task in recent_tasks:
        recent_tasks_data.append({
            'id': task.id,
            'task_id': task.task_id,
            'title': task.title,
            'category': task.category,
            'priority': task.priority,
            'status': task.status,
            'created_at': task.created_at.isoformat() if task.created_at else None
        })
    
    # Category distribution
    from sqlalchemy import func
    category_stats = db.session.query(
        Task.category,
        func.count(Task.id).label('count')
    ).filter(
        Task.created_at >= start_date
    ).group_by(Task.category).all()
    
    category_data = [{'category': cat, 'count': count} for cat, count in category_stats]
    
    # Priority distribution
    priority_stats = db.session.query(
        Task.priority,
        func.count(Task.id).label('count')
    ).filter(
        Task.created_at >= start_date
    ).group_by(Task.priority).all()
    
    priority_data = [{'priority': pri, 'count': count} for pri, count in priority_stats]
    
    # Daily activity
    daily_activity = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=i)
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        created_count = Task.query.filter(
            Task.created_at >= start_of_day,
            Task.created_at <= end_of_day
        ).count()
        
        resolved_count = Task.query.filter(
            Task.status == 'Closed',
            Task.end_time >= start_of_day,
            Task.end_time <= end_of_day
        ).count()
        
        daily_activity.append({
            'date': date.strftime('%Y-%m-%d'),
            'created': created_count,
            'resolved': resolved_count
        })
    
    daily_activity.reverse()  # Oldest to newest
    
    return jsonify({
        'total_tasks': total_tasks,
        'open_tasks': open_tasks,
        'overdue_tasks': overdue_tasks,
        'my_tasks': my_tasks,
        'team_tasks': team_tasks,
        'sla_compliance': round(sla_compliance, 1),
        'recent_tasks': recent_tasks_data,
        'category_distribution': category_data,
        'priority_distribution': priority_data,
        'daily_activity': daily_activity
    })

@dashboard_bp.route('/api/technician-performance')
@login_required
def technician_performance():
    """Get technician performance statistics"""
    if current_user.role not in ['Admin', 'Supervisor']:
        return jsonify({'error': 'Access denied'}), 403
    
    days = int(request.args.get('days', 30))
    start_date = datetime.utcnow() - timedelta(days=days)
    
    technicians = User.query.filter(User.role.in_(['Technician', 'Supervisor'])).all()
    
    performance_data = []
    for tech in technicians:
        # Get assigned tasks
        assigned_tasks = Task.query.filter(
            Task.assigned_to == tech.id,
            Task.created_at >= start_date
        ).all()
        
        completed_tasks = [t for t in assigned_tasks if t.status == 'Closed']
        
        # Calculate metrics
        if completed_tasks:
            total_resolution_time = sum(t.actual_hours or 0 for t in completed_tasks)
            avg_resolution_time = total_resolution_time / len(completed_tasks)
            
            sla_compliant = len([t for t in completed_tasks 
                               if t.sla_due_date and t.end_time and t.end_time <= t.sla_due_date])
            sla_rate = sla_compliant / len(completed_tasks) * 100
        else:
            avg_resolution_time = 0
            sla_rate = 0
        
        performance_data.append({
            'id': tech.id,
            'name': tech.full_name,
            'department': tech.department,
            'assigned_tasks': len(assigned_tasks),
            'completed_tasks': len(completed_tasks),
            'completion_rate': len(completed_tasks) / len(assigned_tasks) * 100 if assigned_tasks else 0,
            'avg_resolution_time': round(avg_resolution_time, 1),
            'sla_compliance_rate': round(sla_rate, 1),
            'current_workload': len([t for t in assigned_tasks if t.status in ['Assigned', 'In Progress']]),
            'is_available': tech.is_available
        })
    
    return jsonify({'technicians': performance_data})

@dashboard_bp.route('/api/department-performance')
@login_required
def department_performance():
    """Get department performance statistics"""
    if current_user.role not in ['Admin', 'Supervisor']:
        return jsonify({'error': 'Access denied'}), 403
    
    days = int(request.args.get('days', 30))
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all departments from tasks
    from sqlalchemy import func
    dept_stats = db.session.query(
        Task.department,
        func.count(Task.id).label('total_tasks'),
        func.sum(func.case((Task.status == 'Closed', 1), else_=0)).label('completed_tasks'),
        func.avg(Task.actual_hours).label('avg_resolution_time')
    ).filter(
        Task.created_at >= start_date
    ).group_by(Task.department).all()
    
    department_data = []
    for dept, total, completed, avg_time in dept_stats:
        department_data.append({
            'department': dept,
            'total_tasks': total,
            'completed_tasks': completed,
            'completion_rate': round(completed / total * 100, 1) if total > 0 else 0,
            'avg_resolution_time': round(avg_time or 0, 1),
            'open_tasks': total - completed
        })
    
    # Sort by completion rate
    department_data.sort(key=lambda x: x['completion_rate'], reverse=True)
    
    return jsonify({'departments': department_data})

@dashboard_bp.route('/api/sla-trends')
@login_required
def sla_trends():
    """Get SLA compliance trends over time"""
    days = int(request.args.get('days', 30))
    
    trend_data = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=i)
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Get tasks completed on this day
        completed_tasks = Task.query.filter(
            Task.status == 'Closed',
            Task.end_time >= start_of_day,
            Task.end_time <= end_of_day
        ).all()
        
        if completed_tasks:
            sla_compliant = len([t for t in completed_tasks 
                               if t.sla_due_date and t.end_time and t.end_time <= t.sla_due_date])
            sla_rate = sla_compliant / len(completed_tasks) * 100
        else:
            sla_rate = 0
        
        trend_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sla_compliance': round(sla_rate, 1),
            'completed_tasks': len(completed_tasks)
        })
    
    trend_data.reverse()  # Oldest to newest
    
    return jsonify({'trends': trend_data})