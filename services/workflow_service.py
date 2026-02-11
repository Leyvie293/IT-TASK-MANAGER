"""
Workflow Service - Handles automated workflows and processes
"""

from datetime import datetime, timedelta
from models.task_models import Task, WorkflowTemplate, TaskActivity
from models.user_models import User
from models.database import db
import json

class WorkflowService:
    """Service for managing automated workflows"""
    
    def __init__(self):
        self.workflow_templates = {}
        self.load_workflow_templates()
    
    def load_workflow_templates(self):
        """Load workflow templates from database"""
        templates = WorkflowTemplate.query.filter_by(is_active=True).all()
        for template in templates:
            self.workflow_templates[template.category] = {
                'id': template.id,
                'name': template.name,
                'steps': template.steps,
                'checklist': template.checklist_items,
                'estimated_time': template.estimated_time,
                'required_skills': template.required_skills
            }
    
    def apply_workflow_to_task(self, task):
        """
        Apply appropriate workflow to a task based on its category
        
        Args:
            task: Task object
            
        Returns:
            Updated task with workflow applied
        """
        category = task.category
        
        # Find matching workflow template
        template = self.workflow_templates.get(category)
        
        if not template:
            # Try to find partial match
            for key, value in self.workflow_templates.items():
                if key in category or category in key:
                    template = value
                    break
        
        if template:
            # Apply workflow steps
            task.workflow_type = template['name']
            task.total_steps = len(template['steps'])
            task.current_step = 1
            
            # Create checklist from template
            if template['checklist']:
                task.checklist = template['checklist']
            
            # Set estimated time if not already set
            if not task.estimated_hours:
                task.estimated_hours = template['estimated_time']
            
            # Log workflow application
            activity = TaskActivity(
                task_id=task.id,
                user_id='system',
                activity_type='Workflow Applied',
                description=f'Applied workflow template: {template["name"]}',
                details={'template_id': template['id']}
            )
            db.session.add(activity)
        
        return task
    
    def get_next_workflow_step(self, task, current_status=None):
        """
        Get the next step in the workflow for a task
        
        Args:
            task: Task object
            current_status: Current status (optional, uses task.status if not provided)
            
        Returns:
            Next workflow step or None if workflow complete
        """
        if not task.workflow_type:
            return None
        
        template = None
        for t in self.workflow_templates.values():
            if t['name'] == task.workflow_type:
                template = t
                break
        
        if not template:
            return None
        
        status = current_status or task.status
        current_step = task.current_step or 1
        
        # Map status to workflow steps
        status_to_step = {
            'New': 1,
            'Acknowledged': 2,
            'Assigned': 3,
            'In Progress': 4,
            'Waiting': 5,
            'Resolved': 6,
            'Closed': 7
        }
        
        step_number = status_to_step.get(status, current_step)
        
        if step_number <= len(template['steps']):
            return template['steps'][step_number - 1]
        
        return None
    
    def advance_workflow(self, task, completed_step=None):
        """
        Advance task to next workflow step
        
        Args:
            task: Task object
            completed_step: Description of completed step
            
        Returns:
            Updated task
        """
        if not task.workflow_type:
            return task
        
        # Update current step
        if task.current_step < task.total_steps:
            task.current_step += 1
        
        # Update progress
        if task.total_steps > 0:
            task.progress = int((task.current_step / task.total_steps) * 100)
        
        # Log workflow advancement
        if completed_step:
            activity = TaskActivity(
                task_id=task.id,
                user_id='system',
                activity_type='Workflow Advanced',
                description=f'Advanced to step {task.current_step}/{task.total_steps}: {completed_step}'
            )
            db.session.add(activity)
        
        return task
    
    def auto_assign_based_on_workflow(self, task):
        """
        Auto-assign task based on workflow requirements
        
        Args:
            task: Task object
            
        Returns:
            Assigned technician or None
        """
        if not task.workflow_type:
            return None
        
        template = None
        for t in self.workflow_templates.values():
            if t['name'] == task.workflow_type:
                template = t
                break
        
        if not template:
            return None
        
        required_skills = template.get('required_skills', [])
        
        if not required_skills:
            return None
        
        # Find technicians with required skills
        technicians = User.query.filter(
            User.role.in_(['Technician', 'Supervisor']),
            User.is_available == True
        ).all()
        
        best_match = None
        best_score = -1
        
        for tech in technicians:
            tech_skills = tech.skills or []
            if isinstance(tech_skills, str):
                tech_skills = [s.strip() for s in tech_skills.split(',')]
            
            # Calculate skill match score
            matching_skills = set(tech_skills) & set(required_skills)
            score = len(matching_skills) / len(required_skills) if required_skills else 0
            
            # Adjust score by workload
            current_tasks = Task.query.filter(
                Task.assigned_to == tech.id,
                Task.status.in_(['Assigned', 'In Progress'])
            ).count()
            
            workload_penalty = current_tasks / (tech.max_tasks or 5)
            adjusted_score = score * (1 - workload_penalty * 0.3)
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_match = tech
        
        if best_match and best_score > 0.3:  # Minimum match threshold
            task.assigned_to = best_match.id
            task.assigned_by = 'system'
            task.assigned_at = datetime.utcnow()
            task.status = 'Assigned'
            
            # Log auto-assignment
            activity = TaskActivity(
                task_id=task.id,
                user_id='system',
                activity_type='Auto-Assignment',
                description=f'Auto-assigned to {best_match.full_name} based on workflow requirements',
                details={
                    'technician_id': best_match.id,
                    'match_score': best_score,
                    'required_skills': required_skills
                }
            )
            db.session.add(activity)
            
            return best_match
        
        return None
    
    def check_and_escalate(self, task):
        """
        Check if task needs escalation based on workflow rules
        
        Args:
            task: Task object
            
        Returns:
            Boolean indicating if escalation was performed
        """
        if not task.workflow_type or not task.assigned_to:
            return False
        
        # Check SLA status
        if task.sla_due_date and datetime.utcnow() > task.sla_due_date:
            return self.escalate_task(task, 'SLA breached')
        
        # Check if task has been in current status too long
        status_duration_map = {
            'New': timedelta(hours=2),
            'Acknowledged': timedelta(hours=1),
            'Assigned': timedelta(hours=4),
            'In Progress': timedelta(hours=24),
            'Waiting': timedelta(hours=48)
        }
        
        if task.status in status_duration_map:
            # Find last status change
            last_status_change = TaskActivity.query.filter(
                TaskActivity.task_id == task.id,
                TaskActivity.activity_type == 'Status Change'
            ).order_by(TaskActivity.created_at.desc()).first()
            
            if last_status_change:
                time_in_status = datetime.utcnow() - last_status_change.created_at
                if time_in_status > status_duration_map[task.status]:
                    return self.escalate_task(task, f'Too long in {task.status} status')
        
        return False
    
    def escalate_task(self, task, reason):
        """
        Escalate a task to supervisor
        
        Args:
            task: Task object
            reason: Reason for escalation
            
        Returns:
            Boolean indicating success
        """
        # Find supervisor in same department
        supervisor = User.query.filter(
            User.department == task.department,
            User.role == 'Supervisor',
            User.is_active == True
        ).first()
        
        if not supervisor:
            # Find any supervisor
            supervisor = User.query.filter(
                User.role == 'Supervisor',
                User.is_active == True
            ).first()
        
        if not supervisor:
            return False
        
        from models.task_models import Escalation
        
        # Create escalation record
        escalation = Escalation(
            task_id=task.id,
            escalated_by='system',
            escalated_to=supervisor.id,
            reason=reason,
            level=1,
            status='Pending'
        )
        db.session.add(escalation)
        
        # Update task
        task.status = 'Escalated'
        
        # Log escalation
        activity = TaskActivity(
            task_id=task.id,
            user_id='system',
            activity_type='Escalation',
            description=f'Task escalated to {supervisor.full_name}: {reason}',
            details={
                'supervisor_id': supervisor.id,
                'reason': reason
            }
        )
        db.session.add(activity)
        
        db.session.commit()
        
        # Send notification
        from .notification_service import NotificationService
        notification_service = NotificationService()
        notification_service.send_task_escalation_notification(task, supervisor, reason)
        
        return True
    
    def generate_workflow_report(self, start_date=None, end_date=None):
        """
        Generate workflow performance report
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Report data
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get tasks with workflows in date range
        tasks = Task.query.filter(
            Task.workflow_type.isnot(None),
            Task.created_at >= start_date,
            Task.created_at <= end_date
        ).all()
        
        report_data = {
            'period': {
                'start': start_date,
                'end': end_date
            },
            'total_tasks': len(tasks),
            'workflow_distribution': {},
            'completion_times': {},
            'escalations': 0,
            'auto_assignments': 0
        }
        
        for task in tasks:
            # Count by workflow type
            workflow_type = task.workflow_type
            report_data['workflow_distribution'][workflow_type] = \
                report_data['workflow_distribution'].get(workflow_type, 0) + 1
            
            # Calculate completion time for closed tasks
            if task.status == 'Closed' and task.start_time and task.end_time:
                completion_time = (task.end_time - task.start_time).total_seconds() / 3600
                if workflow_type not in report_data['completion_times']:
                    report_data['completion_times'][workflow_type] = []
                report_data['completion_times'][workflow_type].append(completion_time)
            
            # Count escalations
            if task.status == 'Escalated':
                report_data['escalations'] += 1
            
            # Check if auto-assigned
            if task.assigned_by == 'system':
                report_data['auto_assignments'] += 1
        
        # Calculate average completion times
        for workflow_type, times in report_data['completion_times'].items():
            if times:
                report_data['completion_times'][workflow_type] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return report_data
    
    def create_custom_workflow(self, name, category, steps, checklist=None, 
                              estimated_time=1.0, required_skills=None):
        """
        Create a custom workflow template
        
        Args:
            name: Workflow name
            category: Task category
            steps: List of workflow steps
            checklist: Optional checklist items
            estimated_time: Estimated completion time in hours
            required_skills: List of required skills
            
        Returns:
            Created workflow template
        """
        template = WorkflowTemplate(
            name=name,
            category=category,
            steps=steps,
            checklist_items=checklist or [],
            estimated_time=estimated_time,
            required_skills=required_skills or [],
            is_active=True
        )
        
        db.session.add(template)
        db.session.commit()
        
        # Reload templates
        self.load_workflow_templates()
        
        return template