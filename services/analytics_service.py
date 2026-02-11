"""
Analytics Service - Handles data analysis and business intelligence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.database import db
from models.task_models import Task, User
from sqlalchemy import func, extract, case, and_
import json
from collections import defaultdict

class AnalyticsService:
    """Service for advanced analytics and reporting"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_performance_metrics(self, start_date=None, end_date=None, department=None):
        """
        Get comprehensive performance metrics
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            department: Optional department filter
            
        Returns:
            Performance metrics dictionary
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        cache_key = f"metrics_{start_date.date()}_{end_date.date()}_{department}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.utcnow() - cached_time).seconds < self.cache_timeout:
                return cached_data
        
        # Build query
        query = db.session.query(Task)
        query = query.filter(
            Task.created_at >= start_date,
            Task.created_at <= end_date
        )
        
        if department:
            query = query.filter(Task.department == department)
        
        tasks = query.all()
        
        if not tasks:
            return {
                'period': {'start': start_date, 'end': end_date},
                'department': department,
                'total_tasks': 0,
                'metrics': {}
            }
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed_tasks = [t for t in tasks if t.status == 'Closed']
        in_progress_tasks = [t for t in tasks if t.status in ['Assigned', 'In Progress']]
        overdue_tasks = [t for t in tasks if t.sla_due_date and t.sla_due_date < datetime.utcnow() 
                        and t.status not in ['Closed', 'Resolved']]
        
        # SLA compliance
        sla_compliant = len([t for t in completed_tasks 
                           if t.sla_due_date and t.end_time and t.end_time <= t.sla_due_date])
        sla_rate = sla_compliant / len(completed_tasks) * 100 if completed_tasks else 0
        
        # Resolution times
        resolution_times = []
        for task in completed_tasks:
            if task.start_time and task.end_time:
                hours = (task.end_time - task.start_time).total_seconds() / 3600
                resolution_times.append(hours)
        
        avg_resolution_time = np.mean(resolution_times) if resolution_times else 0
        median_resolution_time = np.median(resolution_times) if resolution_times else 0
        
        # First response time
        first_response_times = []
        for task in tasks:
            first_activity = TaskActivity.query.filter_by(
                task_id=task.id,
                activity_type='Status Change'
            ).order_by(TaskActivity.created_at).first()
            
            if first_activity and task.created_at:
                hours = (first_activity.created_at - task.created_at).total_seconds() / 3600
                first_response_times.append(hours)
        
        avg_first_response = np.mean(first_response_times) if first_response_times else 0
        
        # Technician performance
        technician_stats = self._get_technician_performance(start_date, end_date, department)
        
        # Category analysis
        category_stats = self._get_category_analysis(start_date, end_date, department)
        
        # Priority analysis
        priority_stats = self._get_priority_analysis(start_date, end_date, department)
        
        # Trend analysis
        trends = self._get_trend_analysis(start_date, end_date, department)
        
        metrics = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'department': department,
            'total_tasks': total_tasks,
            'completed_tasks': len(completed_tasks),
            'in_progress_tasks': len(in_progress_tasks),
            'overdue_tasks': len(overdue_tasks),
            'completion_rate': len(completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0,
            'sla_compliance_rate': sla_rate,
            'avg_resolution_time': avg_resolution_time,
            'median_resolution_time': median_resolution_time,
            'avg_first_response_time': avg_first_response,
            'technician_performance': technician_stats,
            'category_analysis': category_stats,
            'priority_analysis': priority_stats,
            'trends': trends,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        # Cache results
        self.cache[cache_key] = (metrics, datetime.utcnow())
        
        return metrics
    
    def _get_technician_performance(self, start_date, end_date, department=None):
        """Get technician performance statistics"""
        query = db.session.query(
            User.id,
            User.full_name,
            User.department,
            func.count(Task.id).label('total_tasks'),
            func.sum(case((Task.status == 'Closed', 1), else_=0)).label('completed_tasks'),
            func.avg(Task.actual_hours).label('avg_resolution_time'),
            func.sum(case((and_(
                Task.status == 'Closed',
                Task.sla_due_date.isnot(None),
                Task.end_time <= Task.sla_due_date
            ), 1), else_=0)).label('sla_compliant_tasks')
        ).join(
            Task, User.id == Task.assigned_to
        ).filter(
            Task.created_at >= start_date,
            Task.created_at <= end_date,
            User.role.in_(['Technician', 'Supervisor'])
        )
        
        if department:
            query = query.filter(User.department == department)
        
        query = query.group_by(User.id, User.full_name, User.department)
        results = query.all()
        
        technician_stats = []
        for tech in results:
            tech_id, name, dept, total, completed, avg_time, sla_compliant = tech
            
            if completed > 0:
                completion_rate = completed / total * 100
                sla_rate = sla_compliant / completed * 100 if completed > 0 else 0
            else:
                completion_rate = 0
                sla_rate = 0
            
            technician_stats.append({
                'id': tech_id,
                'name': name,
                'department': dept,
                'total_tasks': total,
                'completed_tasks': completed,
                'completion_rate': round(completion_rate, 1),
                'avg_resolution_time': round(avg_time or 0, 1),
                'sla_compliance_rate': round(sla_rate, 1),
                'sla_compliant_tasks': sla_compliant
            })
        
        return technician_stats
    
    def _get_category_analysis(self, start_date, end_date, department=None):
        """Analyze tasks by category"""
        query = db.session.query(
            Task.category,
            func.count(Task.id).label('total'),
            func.sum(case((Task.status == 'Closed', 1), else_=0)).label('completed'),
            func.avg(Task.actual_hours).label('avg_time'),
            func.avg(case((
                and_(
                    Task.status == 'Closed',
                    Task.sla_due_date.isnot(None),
                    Task.end_time <= Task.sla_due_date
                ), 1.0
            ), else_=0.0)).label('sla_rate')
        ).filter(
            Task.created_at >= start_date,
            Task.created_at <= end_date
        )
        
        if department:
            query = query.filter(Task.department == department)
        
        query = query.group_by(Task.category)
        results = query.all()
        
        category_stats = []
        for category, total, completed, avg_time, sla_rate in results:
            category_stats.append({
                'category': category,
                'total_tasks': total,
                'completed_tasks': completed,
                'completion_rate': round(completed / total * 100, 1) if total > 0 else 0,
                'avg_resolution_time': round(avg_time or 0, 1),
                'sla_compliance_rate': round((sla_rate or 0) * 100, 1)
            })
        
        return category_stats
    
    def _get_priority_analysis(self, start_date, end_date, department=None):
        """Analyze tasks by priority"""
        query = db.session.query(
            Task.priority,
            func.count(Task.id).label('total'),
            func.sum(case((Task.status == 'Closed', 1), else_=0)).label('completed'),
            func.avg(Task.actual_hours).label('avg_time'),
            func.sum(case((and_(
                Task.sla_due_date.isnot(None),
                Task.sla_due_date < datetime.utcnow(),
                Task.status.notin_(['Closed', 'Resolved'])
            ), 1), else_=0)).label('overdue')
        ).filter(
            Task.created_at >= start_date,
            Task.created_at <= end_date
        )
        
        if department:
            query = query.filter(Task.department == department)
        
        query = query.group_by(Task.priority)
        results = query.all()
        
        priority_stats = []
        for priority, total, completed, avg_time, overdue in results:
            priority_stats.append({
                'priority': priority,
                'total_tasks': total,
                'completed_tasks': completed,
                'completion_rate': round(completed / total * 100, 1) if total > 0 else 0,
                'avg_resolution_time': round(avg_time or 0, 1),
                'overdue_tasks': overdue,
                'overdue_rate': round(overdue / total * 100, 1) if total > 0 else 0
            })
        
        return priority_stats
    
    def _get_trend_analysis(self, start_date, end_date, department=None):
        """Analyze trends over time"""
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        trends = []
        for date in date_range:
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            query = db.session.query(
                func.count(Task.id).label('created'),
                func.sum(case((Task.status == 'Closed', 1), else_=0)).label('completed'),
                func.avg(Task.actual_hours).label('avg_time')
            ).filter(
                Task.created_at >= day_start,
                Task.created_at <= day_end
            )
            
            if department:
                query = query.filter(Task.department == department)
            
            result = query.first()
            created, completed, avg_time = result
            
            trends.append({
                'date': date.strftime('%Y-%m-%d'),
                'tasks_created': created or 0,
                'tasks_completed': completed or 0,
                'avg_resolution_time': round(avg_time or 0, 1)
            })
        
        return trends
    
    def predict_future_workload(self, days_ahead=7):
        """
        Predict future workload based on historical patterns
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            Predicted workload data
        """
        # Get historical data (last 90 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)
        
        # Group by day of week and hour
        daily_patterns = defaultdict(list)
        hourly_patterns = defaultdict(lambda: defaultdict(int))
        
        tasks = Task.query.filter(
            Task.created_at >= start_date,
            Task.created_at <= end_date
        ).all()
        
        for task in tasks:
            created_date = task.created_at
            day_of_week = created_date.weekday()  # 0 = Monday, 6 = Sunday
            hour_of_day = created_date.hour
            
            daily_patterns[day_of_week].append(1)
            hourly_patterns[day_of_week][hour_of_day] += 1
        
        # Calculate averages
        avg_daily = {}
        for day in range(7):
            if day in daily_patterns:
                avg_daily[day] = np.mean(daily_patterns[day])
            else:
                avg_daily[day] = 0
        
        avg_hourly = {}
        for day in range(7):
            avg_hourly[day] = {}
            for hour in range(24):
                if day in hourly_patterns and hour in hourly_patterns[day]:
                    avg_hourly[day][hour] = hourly_patterns[day][hour] / len(daily_patterns.get(day, [1]))
                else:
                    avg_hourly[day][hour] = 0
        
        # Generate predictions
        predictions = []
        for i in range(days_ahead):
            date = end_date + timedelta(days=i+1)
            day_of_week = date.weekday()
            
            predicted_tasks = avg_daily.get(day_of_week, 0)
            
            # Adjust for seasonality (month)
            month = date.month
            month_factor = self._get_monthly_factor(month)
            predicted_tasks *= month_factor
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': day_of_week,
                'predicted_tasks': round(predicted_tasks, 1),
                'hourly_distribution': avg_hourly.get(day_of_week, {})
            })
        
        return {
            'prediction_period': days_ahead,
            'historical_period_days': 90,
            'predictions': predictions,
            'avg_daily_tasks': np.mean(list(avg_daily.values())),
            'confidence_level': self._calculate_confidence(daily_patterns)
        }
    
    def _get_monthly_factor(self, month):
        """Get monthly adjustment factor based on historical patterns"""
        # These factors should be calculated from historical data
        # For now, using estimated factors
        monthly_factors = {
            1: 0.9,   # January
            2: 0.95,  # February
            3: 1.0,   # March
            4: 1.05,  # April
            5: 1.1,   # May
            6: 1.05,  # June
            7: 1.0,   # July
            8: 0.95,  # August
            9: 1.0,   # September
            10: 1.1,  # October
            11: 1.05, # November
            12: 0.9   # December
        }
        
        return monthly_factors.get(month, 1.0)
    
    def _calculate_confidence(self, patterns):
        """Calculate confidence level for predictions"""
        if not patterns:
            return 0.0
        
        # Calculate coefficient of variation
        all_counts = []
        for day_counts in patterns.values():
            all_counts.extend(day_counts)
        
        if len(all_counts) < 2:
            return 0.5
        
        cv = np.std(all_counts) / np.mean(all_counts)
        
        # Convert to confidence score (lower CV = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - cv))
        
        return round(confidence, 2)
    
    def get_bottleneck_analysis(self, days=30):
        """
        Identify bottlenecks in the workflow
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Bottleneck analysis results
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Analyze time spent in each status
        status_times = defaultdict(list)
        
        tasks = Task.query.filter(
            Task.status == 'Closed',
            Task.created_at >= start_date
        ).all()
        
        for task in tasks:
            # Get status change activities
            activities = TaskActivity.query.filter(
                TaskActivity.task_id == task.id,
                TaskActivity.activity_type == 'Status Change'
            ).order_by(TaskActivity.created_at).all()
            
            if len(activities) >= 2:
                for i in range(len(activities) - 1):
                    status = activities[i].new_value
                    duration = (activities[i + 1].created_at - activities[i].created_at).total_seconds() / 3600
                    status_times[status].append(duration)
        
        # Calculate statistics for each status
        bottlenecks = []
        for status, times in status_times.items():
            if times:
                bottlenecks.append({
                    'status': status,
                    'avg_hours': np.mean(times),
                    'median_hours': np.median(times),
                    'max_hours': max(times),
                    'task_count': len(times),
                    'total_hours': sum(times),
                    'is_bottleneck': np.mean(times) > 24  # Flag if average > 24 hours
                })
        
        # Sort by average time (descending)
        bottlenecks.sort(key=lambda x: x['avg_hours'], reverse=True)
        
        # Calculate overall workflow efficiency
        if bottlenecks:
            total_time = sum(b['total_hours'] for b in bottlenecks)
            bottleneck_time = sum(b['total_hours'] for b in bottlenecks if b['is_bottleneck'])
            efficiency = 1 - (bottleneck_time / total_time) if total_time > 0 else 1.0
        else:
            efficiency = 1.0
        
        return {
            'analysis_period_days': days,
            'bottlenecks': bottlenecks,
            'workflow_efficiency': round(efficiency * 100, 1),
            'recommendations': self._generate_bottleneck_recommendations(bottlenecks)
        }
    
    def _generate_bottleneck_recommendations(self, bottlenecks):
        """Generate recommendations for addressing bottlenecks"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['is_bottleneck']:
                status = bottleneck['status']
                avg_hours = bottleneck['avg_hours']
                
                if status == 'New':
                    recommendations.append(
                        f"Tasks stay in 'New' status for {avg_hours:.1f} hours on average. "
                        "Consider implementing auto-acknowledgment or reducing initial review time."
                    )
                elif status == 'Assigned':
                    recommendations.append(
                        f"Tasks wait {avg_hours:.1f} hours after assignment before work begins. "
                        "Review technician workload distribution or implement assignment notifications."
                    )
                elif status == 'Waiting':
                    recommendations.append(
                        f"Tasks spend {avg_hours:.1f} hours in 'Waiting' status. "
                        "Improve vendor response times or implement escalation procedures."
                    )
                elif status == 'In Progress':
                    recommendations.append(
                        f"Tasks take {avg_hours:.1f} hours in 'In Progress' status. "
                        "Consider breaking down complex tasks or providing additional resources."
                    )
        
        if not recommendations:
            recommendations.append("No significant bottlenecks detected. Workflow is efficient.")
        
        return recommendations
    
    def export_analytics_data(self, start_date, end_date, format='json'):
        """
        Export analytics data in specified format
        
        Args:
            start_date: Start date
            end_date: End date
            format: Export format (json, csv, excel)
            
        Returns:
            File path or data
        """
        metrics = self.get_performance_metrics(start_date, end_date)
        
        if format == 'json':
            return json.dumps(metrics, indent=2, default=str)
        
        elif format == 'csv':
            # Flatten data for CSV
            csv_data = []
            
            # Add summary metrics
            csv_data.append(['Metric', 'Value'])
            csv_data.append(['Total Tasks', metrics['total_tasks']])
            csv_data.append(['Completed Tasks', metrics['completed_tasks']])
            csv_data.append(['SLA Compliance Rate', f"{metrics['sla_compliance_rate']:.1f}%"])
            csv_data.append(['Average Resolution Time', f"{metrics['avg_resolution_time']:.1f} hours"])
            csv_data.append(['', ''])
            
            # Add technician performance
            csv_data.append(['Technician Performance', '', '', '', ''])
            csv_data.append(['Name', 'Department', 'Total Tasks', 'Completion Rate', 'SLA Rate'])
            for tech in metrics['technician_performance']:
                csv_data.append([
                    tech['name'],
                    tech['department'],
                    tech['total_tasks'],
                    f"{tech['completion_rate']:.1f}%",
                    f"{tech['sla_compliance_rate']:.1f}%"
                ])
            
            # Convert to CSV string
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            writer.writerows(csv_data)
            
            return output.getvalue()
        
        elif format == 'excel':
            import pandas as pd
            from io import BytesIO
            
            # Create Excel file with multiple sheets
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([
                    ['Total Tasks', metrics['total_tasks']],
                    ['Completed Tasks', metrics['completed_tasks']],
                    ['SLA Compliance Rate', f"{metrics['sla_compliance_rate']:.1f}%"],
                    ['Average Resolution Time', f"{metrics['avg_resolution_time']:.1f} hours"]
                ], columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Technician sheet
                if metrics['technician_performance']:
                    tech_df = pd.DataFrame(metrics['technician_performance'])
                    tech_df.to_excel(writer, sheet_name='Technicians', index=False)
                
                # Category sheet
                if metrics['category_analysis']:
                    cat_df = pd.DataFrame(metrics['category_analysis'])
                    cat_df.to_excel(writer, sheet_name='Categories', index=False)
                
                # Trends sheet
                if metrics['trends']:
                    trend_df = pd.DataFrame(metrics['trends'])
                    trend_df.to_excel(writer, sheet_name='Trends', index=False)
            
            output.seek(0)
            return output
        
        return None