"""
Report Service - Handles report generation and distribution
"""

import os
import json
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import pandas as pd
from io import BytesIO
from flask import current_app
from models.database import db
from models.task_models import Task, User
from models.report_models import Report
from sqlalchemy import func, extract

class ReportService:
    """Service for generating reports in various formats"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for reports"""
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))
    
    def generate_sla_report(self, start_date, end_date, department=None, format='pdf'):
        """
        Generate SLA compliance report
        
        Args:
            start_date: Start date
            end_date: End date
            department: Optional department filter
            format: Output format (pdf, excel, csv)
            
        Returns:
            Report file or data
        """
        # Get data
        from .analytics_service import AnalyticsService
        analytics = AnalyticsService()
        metrics = analytics.get_performance_metrics(start_date, end_date, department)
        
        if format == 'pdf':
            return self._generate_sla_pdf(metrics)
        elif format == 'excel':
            return self._generate_sla_excel(metrics)
        elif format == 'csv':
            return self._generate_sla_csv(metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_sla_pdf(self, metrics):
        """Generate SLA report in PDF format"""
        buffer = BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        
        # Title
        title = Paragraph("SLA Compliance Report", self.styles['ReportTitle'])
        elements.append(title)
        
        # Period
        period_text = f"Period: {metrics['period']['start']} to {metrics['period']['end']}"
        if metrics['department']:
            period_text += f" | Department: {metrics['department']}"
        
        period = Paragraph(period_text, self.styles['BodyText'])
        elements.append(period)
        
        elements.append(Spacer(1, 20))
        
        # Summary Table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tasks', metrics['total_tasks']],
            ['Completed Tasks', metrics['completed_tasks']],
            ['In Progress Tasks', metrics['in_progress_tasks']],
            ['Overdue Tasks', metrics['overdue_tasks']],
            ['Completion Rate', f"{metrics['completion_rate']:.1f}%"],
            ['SLA Compliance Rate', f"{metrics['sla_compliance_rate']:.1f}%"],
            ['Average Resolution Time', f"{metrics['avg_resolution_time']:.1f} hours"],
            ['Average First Response', f"{metrics['avg_first_response_time']:.1f} hours"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Technician Performance
        tech_title = Paragraph("Technician Performance", self.styles['SectionTitle'])
        elements.append(tech_title)
        
        if metrics['technician_performance']:
            tech_headers = ['Name', 'Department', 'Total Tasks', 'Completed', 'Completion Rate', 'SLA Rate', 'Avg Time']
            tech_rows = [tech_headers]
            
            for tech in metrics['technician_performance']:
                tech_rows.append([
                    tech['name'],
                    tech['department'],
                    str(tech['total_tasks']),
                    str(tech['completed_tasks']),
                    f"{tech['completion_rate']:.1f}%",
                    f"{tech['sla_compliance_rate']:.1f}%",
                    f"{tech['avg_resolution_time']:.1f}h"
                ])
            
            tech_table = Table(tech_rows, repeatRows=1)
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            elements.append(tech_table)
            elements.append(Spacer(1, 20))
        
        # Category Analysis
        cat_title = Paragraph("Category Analysis", self.styles['SectionTitle'])
        elements.append(cat_title)
        
        if metrics['category_analysis']:
            cat_headers = ['Category', 'Total Tasks', 'Completed', 'Completion Rate', 'Avg Time', 'SLA Rate']
            cat_rows = [cat_headers]
            
            for cat in metrics['category_analysis']:
                cat_rows.append([
                    cat['category'],
                    str(cat['total_tasks']),
                    str(cat['completed_tasks']),
                    f"{cat['completion_rate']:.1f}%",
                    f"{cat['avg_resolution_time']:.1f}h",
                    f"{cat['sla_compliance_rate']:.1f}%"
                ])
            
            cat_table = Table(cat_rows, repeatRows=1)
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            elements.append(cat_table)
        
        # Footer
        elements.append(Spacer(1, 20))
        generated = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            self.styles['BodyText'])
        elements.append(generated)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return buffer
    
    def _generate_sla_excel(self, metrics):
        """Generate SLA report in Excel format"""
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([
                ['Period', f"{metrics['period']['start']} to {metrics['period']['end']}"],
                ['Department', metrics['department'] or 'All'],
                ['Total Tasks', metrics['total_tasks']],
                ['Completed Tasks', metrics['completed_tasks']],
                ['Completion Rate', f"{metrics['completion_rate']:.1f}%"],
                ['SLA Compliance Rate', f"{metrics['sla_compliance_rate']:.1f}%"],
                ['Average Resolution Time', f"{metrics['avg_resolution_time']:.1f} hours"],
                ['Average First Response', f"{metrics['avg_first_response_time']:.1f} hours"]
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
            
            # Priority sheet
            if 'priority_analysis' in metrics and metrics['priority_analysis']:
                pri_df = pd.DataFrame(metrics['priority_analysis'])
                pri_df.to_excel(writer, sheet_name='Priorities', index=False)
            
            # Trends sheet
            if 'trends' in metrics and metrics['trends']:
                trend_df = pd.DataFrame(metrics['trends'])
                trend_df.to_excel(writer, sheet_name='Trends', index=False)
        
        buffer.seek(0)
        return buffer
    
    def _generate_sla_csv(self, metrics):
        """Generate SLA report in CSV format"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['SLA Compliance Report'])
        writer.writerow([f"Period: {metrics['period']['start']} to {metrics['period']['end']}"])
        if metrics['department']:
            writer.writerow([f"Department: {metrics['department']}"])
        writer.writerow([])
        
        # Write summary
        writer.writerow(['Summary'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Tasks', metrics['total_tasks']])
        writer.writerow(['Completed Tasks', metrics['completed_tasks']])
        writer.writerow(['Completion Rate', f"{metrics['completion_rate']:.1f}%"])
        writer.writerow(['SLA Compliance Rate', f"{metrics['sla_compliance_rate']:.1f}%"])
        writer.writerow([])
        
        # Write technician performance
        if metrics['technician_performance']:
            writer.writerow(['Technician Performance'])
            writer.writerow(['Name', 'Department', 'Total Tasks', 'Completed', 'Completion Rate', 'SLA Rate', 'Avg Time'])
            for tech in metrics['technician_performance']:
                writer.writerow([
                    tech['name'],
                    tech['department'],
                    tech['total_tasks'],
                    tech['completed_tasks'],
                    f"{tech['completion_rate']:.1f}%",
                    f"{tech['sla_compliance_rate']:.1f}%",
                    f"{tech['avg_resolution_time']:.1f}"
                ])
            writer.writerow([])
        
        output.seek(0)
        return output.getvalue()
    
    def generate_technician_report(self, technician_id, start_date, end_date, format='pdf'):
        """
        Generate individual technician performance report
        
        Args:
            technician_id: Technician user ID
            start_date: Start date
            end_date: End date
            format: Output format
            
        Returns:
            Report file or data
        """
        # Get technician data
        technician = User.query.get(technician_id)
        if not technician:
            raise ValueError("Technician not found")
        
        # Get technician's tasks
        tasks = Task.query.filter(
            Task.assigned_to == technician_id,
            Task.created_at >= start_date,
            Task.created_at <= end_date
        ).all()
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed_tasks = [t for t in tasks if t.status == 'Closed']
        in_progress_tasks = [t for t in tasks if t.status in ['Assigned', 'In Progress']]
        
        # Resolution times
        resolution_times = []
        for task in completed_tasks:
            if task.start_time and task.end_time:
                hours = (task.end_time - task.start_time).total_seconds() / 3600
                resolution_times.append(hours)
        
        avg_resolution_time = np.mean(resolution_times) if resolution_times else 0
        
        # SLA compliance
        sla_compliant = len([t for t in completed_tasks 
                           if t.sla_due_date and t.end_time and t.end_time <= t.sla_due_date])
        sla_rate = sla_compliant / len(completed_tasks) * 100 if completed_tasks else 0
        
        # Category breakdown
        category_stats = {}
        for task in tasks:
            category = task.category
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'completed': 0}
            category_stats[category]['total'] += 1
            if task.status == 'Closed':
                category_stats[category]['completed'] += 1
        
        # Prepare report data
        report_data = {
            'technician': {
                'id': technician.id,
                'name': technician.full_name,
                'department': technician.department,
                'email': technician.email,
                'job_title': technician.job_title,
                'skills': technician.skills or []
            },
            'period': {
                'start': start_date,
                'end': end_date
            },
            'metrics': {
                'total_tasks': total_tasks,
                'completed_tasks': len(completed_tasks),
                'in_progress_tasks': len(in_progress_tasks),
                'completion_rate': len(completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0,
                'sla_compliance_rate': sla_rate,
                'avg_resolution_time': avg_resolution_time,
                'avg_tasks_per_day': total_tasks / max((end_date - start_date).days, 1)
            },
            'category_breakdown': category_stats,
            'recent_tasks': [
                {
                    'task_id': t.task_id,
                    'title': t.title,
                    'category': t.category,
                    'priority': t.priority,
                    'status': t.status,
                    'created_at': t.created_at.isoformat() if t.created_at else None,
                    'completed_at': t.end_time.isoformat() if t.end_time else None,
                    'resolution_time': (t.end_time - t.start_time).total_seconds() / 3600 
                                      if t.start_time and t.end_time else None
                }
                for t in completed_tasks[:10]  # Last 10 completed tasks
            ]
        }
        
        if format == 'pdf':
            return self._generate_technician_pdf(report_data)
        elif format == 'excel':
            return self._generate_technician_excel(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_technician_pdf(self, data):
        """Generate technician report in PDF format"""
        buffer = BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        
        # Title
        title = Paragraph(f"Technician Performance Report: {data['technician']['name']}", 
                         self.styles['ReportTitle'])
        elements.append(title)
        
        # Technician Info
        tech_info = [
            ['Name:', data['technician']['name']],
            ['Department:', data['technician']['department']],
            ['Job Title:', data['technician']['job_title'] or 'Not specified'],
            ['Email:', data['technician']['email']],
            ['Skills:', ', '.join(data['technician']['skills']) if data['technician']['skills'] else 'Not specified'],
            ['Report Period:', f"{data['period']['start'].strftime('%Y-%m-%d')} to {data['period']['end'].strftime('%Y-%m-%d')}"]
        ]
        
        info_table = Table(tech_info, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Performance Metrics
        metrics_title = Paragraph("Performance Metrics", self.styles['SectionTitle'])
        elements.append(metrics_title)
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Tasks Assigned', data['metrics']['total_tasks']],
            ['Tasks Completed', data['metrics']['completed_tasks']],
            ['Tasks In Progress', data['metrics']['in_progress_tasks']],
            ['Completion Rate', f"{data['metrics']['completion_rate']:.1f}%"],
            ['SLA Compliance Rate', f"{data['metrics']['sla_compliance_rate']:.1f}%"],
            ['Average Resolution Time', f"{data['metrics']['avg_resolution_time']:.1f} hours"],
            ['Average Tasks Per Day', f"{data['metrics']['avg_tasks_per_day']:.1f}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
        
        # Category Breakdown
        if data['category_breakdown']:
            cat_title = Paragraph("Category Breakdown", self.styles['SectionTitle'])
            elements.append(cat_title)
            
            cat_headers = ['Category', 'Total Tasks', 'Completed', 'Completion Rate']
            cat_rows = [cat_headers]
            
            for category, stats in data['category_breakdown'].items():
                completion_rate = stats['completed'] / stats['total'] * 100 if stats['total'] > 0 else 0
                cat_rows.append([
                    category,
                    str(stats['total']),
                    str(stats['completed']),
                    f"{completion_rate:.1f}%"
                ])
            
            cat_table = Table(cat_rows, repeatRows=1)
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            
            elements.append(cat_table)
            elements.append(Spacer(1, 20))
        
        # Recent Tasks
        if data['recent_tasks']:
            recent_title = Paragraph("Recent Completed Tasks", self.styles['SectionTitle'])
            elements.append(recent_title)
            
            recent_headers = ['Task ID', 'Title', 'Category', 'Priority', 'Resolution Time']
            recent_rows = [recent_headers]
            
            for task in data['recent_tasks']:
                resolution_time = task['resolution_time']
                if resolution_time:
                    time_str = f"{resolution_time:.1f}h"
                else:
                    time_str = 'N/A'
                
                recent_rows.append([
                    task['task_id'],
                    task['title'][:30] + '...' if len(task['title']) > 30 else task['title'],
                    task['category'],
                    task['priority'],
                    time_str
                ])
            
            recent_table = Table(recent_rows, repeatRows=1, colWidths=[1*inch, 2.5*inch, 1*inch, 0.8*inch, 1*inch])
            recent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (1, 1), (1, -1), 'LEFT')
            ]))
            
            elements.append(recent_table)
        
        # Footer
        elements.append(Spacer(1, 20))
        generated = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            self.styles['BodyText'])
        elements.append(generated)
        
        # Recommendations
        recommendations = self._generate_technician_recommendations(data)
        if recommendations:
            elements.append(Spacer(1, 10))
            rec_title = Paragraph("Recommendations", self.styles['SectionTitle'])
            elements.append(rec_title)
            
            for rec in recommendations:
                rec_para = Paragraph(f"• {rec}", self.styles['BodyText'])
                elements.append(rec_para)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return buffer
    
    def _generate_technician_recommendations(self, data):
        """Generate recommendations for technician improvement"""
        recommendations = []
        metrics = data['metrics']
        
        if metrics['sla_compliance_rate'] < 80:
            recommendations.append(
                "Focus on improving SLA compliance. Prioritize tasks with approaching deadlines."
            )
        
        if metrics['avg_resolution_time'] > 24:
            recommendations.append(
                "Work on reducing resolution times. Consider breaking down complex tasks."
            )
        
        if metrics['completion_rate'] < 70:
            recommendations.append(
                "Improve task completion rate. Avoid taking on too many tasks simultaneously."
            )
        
        # Check category performance
        for category, stats in data['category_breakdown'].items():
            if stats['total'] > 5:  # Only for categories with significant tasks
                completion_rate = stats['completed'] / stats['total'] * 100
                if completion_rate < 60:
                    recommendations.append(
                        f"Seek additional training or support for {category} tasks."
                    )
        
        if not recommendations:
            recommendations.append("Continue current performance. No specific recommendations at this time.")
        
        return recommendations
    
    def save_report(self, report_type, parameters, generated_by, file_path, format):
        """
        Save report metadata to database
        
        Args:
            report_type: Type of report
            parameters: Report parameters
            generated_by: User ID who generated the report
            file_path: Path to saved report file
            format: Report format
            
        Returns:
            Report object
        """
        report = Report(
            report_id=f"REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            name=f"{report_type} Report - {datetime.now().strftime('%Y-%m-%d')}",
            report_type=report_type,
            parameters=parameters,
            generated_by=generated_by,
            file_path=file_path,
            format=format
        )
        
        db.session.add(report)
        db.session.commit()
        
        return report
    
    def get_report_history(self, user_id=None, report_type=None, limit=50):
        """
        Get report generation history
        
        Args:
            user_id: Filter by user
            report_type: Filter by report type
            limit: Maximum number of reports to return
            
        Returns:
            List of reports
        """
        query = Report.query
        
        if user_id:
            query = query.filter(Report.generated_by == user_id)
        
        if report_type:
            query = query.filter(Report.report_type == report_type)
        
        reports = query.order_by(Report.created_at.desc()).limit(limit).all()
        
        return [r.to_dict() for r in reports]