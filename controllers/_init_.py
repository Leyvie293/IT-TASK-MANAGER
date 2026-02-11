"""
Controllers package for Riley Falcon I.T Task Manager
"""

from .auth_controller import auth_bp
from .task_controller import task_bp
from .report_controller import report_bp
from .admin_controller import admin_bp
from .dashboard_controller import dashboard_bp

__all__ = ['auth_bp', 'task_bp', 'report_bp', 'admin_bp', 'dashboard_bp']