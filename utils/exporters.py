"""
Export utilities for the I.T Task Manager
Comprehensive export functionality for tasks, projects, and reports
"""

import csv
import json
from io import StringIO, BytesIO
from datetime import datetime, date
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import KeepTogether
import os


class TaskExporter:
    """Main exporter class for task management system"""
    
    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], filename: str = None, 
                     delimiter: str = ',', include_headers: bool = True) -> Union[str, str]:
        """
        Export data to CSV format
        
        Args:
            data: List of dictionaries
            filename: Output filename (optional)
            delimiter: Column delimiter
            include_headers: Whether to include header row
            
        Returns:
            CSV string or filename
        """
        if not data:
            return "" if filename is None else filename
        
        output = StringIO()
        
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers, delimiter=delimiter)
            
            if include_headers:
                writer.writeheader()
            
            for row in data:
                # Convert any non-string values
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, (date, datetime)):
                        processed_row[key] = value.isoformat()
                    elif value is None:
                        processed_row[key] = ''
                    else:
                        processed_row[key] = str(value)
                writer.writerow(processed_row)
        else:
            writer = csv.writer(output, delimiter=delimiter)
            if include_headers and data:
                writer.writerow(data[0])
                writer.writerows(data[1:])
            else:
                writer.writerows(data)
        
        csv_data = output.getvalue()
        output.close()
        
        if filename:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_data)
            return filename
        
        return csv_data
    
    @staticmethod
    def export_to_excel(data: Union[List[Dict[str, Any]], pd.DataFrame], 
                       sheet_name: str = 'Data', filename: str = None,
                       auto_adjust_columns: bool = True) -> Union[bytes, str]:
        """
        Export data to Excel format
        
        Args:
            data: List of dictionaries or DataFrame
            sheet_name: Sheet name
            filename: Output filename (optional)
            auto_adjust_columns: Auto-adjust column widths
            
        Returns:
            Excel bytes or filename
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if auto_adjust_columns:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        excel_data = output.getvalue()
        output.close()
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(excel_data)
            return filename
        
        return excel_data
    
    @staticmethod
    def export_to_pdf(data: List[Dict[str, Any]], title: str = "Report", 
                     columns: List[tuple] = None, filename: str = None,
                     orientation: str = 'portrait', include_summary: bool = True,
                     group_by: str = None) -> Union[bytes, str]:
        """
        Export data to PDF format
        
        Args:
            data: List of dictionaries
            title: Report title
            columns: Column definitions (list of tuples: (header, field_name, width))
            filename: Output filename (optional)
            orientation: 'portrait' or 'landscape'
            include_summary: Include summary statistics
            group_by: Field to group data by
            
        Returns:
            PDF bytes or filename
        """
        if not data:
            raise ValueError("No data to export")
        
        buffer = BytesIO()
        
        # Choose page size and orientation
        pagesize = A4 if orientation == 'portrait' else (A4[1], A4[0])
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=pagesize,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            alignment=1  # Center
        )
        
        subheader_style = ParagraphStyle(
            'CustomSubheader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.HexColor('#2E5A88')
        )
        
        # Title
        title_para = Paragraph(title, header_style)
        elements.append(title_para)
        
        # Timestamp
        timestamp = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Italic']
        )
        elements.append(timestamp)
        elements.append(Spacer(1, 20))
        
        # Summary statistics
        if include_summary:
            summary_data = [
                ['Total Records', str(len(data))],
                ['Export Date', datetime.now().strftime('%Y-%m-%d')],
                ['Export Time', datetime.now().strftime('%H:%M:%S')],
                ['Data Fields', str(len(data[0])) if data else '0']
            ]
            
            # Add status summary if data has status field
            if data and 'status' in data[0]:
                status_counts = {}
                for item in data:
                    status = item.get('status', 'Unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                for status, count in status_counts.items():
                    summary_data.append([f'Status: {status}', str(count)])
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D9E2F3')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            elements.append(Paragraph("Summary", subheader_style))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
        
        # Group data if requested
        if group_by and data and group_by in data[0]:
            grouped_data = {}
            for item in data:
                group_value = item.get(group_by, 'Ungrouped')
                if group_value not in grouped_data:
                    grouped_data[group_value] = []
                grouped_data[group_value].append(item)
            
            for group_value, group_items in grouped_data.items():
                elements.append(PageBreak() if elements else None)
                elements.append(Paragraph(f"Group: {group_value}", subheader_style))
                elements.append(Spacer(1, 10))
                
                table_data = TaskExporter._prepare_table_data(group_items, columns)
                table = TaskExporter._create_pdf_table(table_data, columns)
                elements.append(table)
                elements.append(Spacer(1, 10))
        else:
            # Main data table
            elements.append(Paragraph("Data Details", subheader_style))
            elements.append(Spacer(1, 10))
            
            table_data = TaskExporter._prepare_table_data(data, columns)
            table = TaskExporter._create_pdf_table(table_data, columns)
            elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        pdf_data = buffer.getvalue()
        buffer.close()
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(pdf_data)
            return filename
        
        return pdf_data
    
    @staticmethod
    def _prepare_table_data(data: List[Dict[str, Any]], columns: List[tuple]) -> List[List[str]]:
        """Prepare table data for PDF export"""
        if not data:
            return [[]]
        
        if columns:
            headers = [col[0] for col in columns]
            field_names = [col[1] for col in columns]
        else:
            # Use all fields from first data item
            headers = list(data[0].keys())
            field_names = headers
        
        table_data = [headers]
        
        for row in data:
            row_data = []
            for field in field_names:
                value = row.get(field, '')
                if isinstance(value, (date, datetime)):
                    row_data.append(value.strftime('%Y-%m-%d %H:%M'))
                elif isinstance(value, bool):
                    row_data.append('Yes' if value else 'No')
                elif value is None:
                    row_data.append('')
                else:
                    # Truncate long text for PDF display
                    text = str(value)
                    if len(text) > 100:
                        row_data.append(text[:97] + '...')
                    else:
                        row_data.append(text)
            table_data.append(row_data)
        
        return table_data
    
    @staticmethod
    def _create_pdf_table(table_data: List[List[str]], columns: List[tuple]) -> Table:
        """Create styled PDF table"""
        if columns:
            col_widths = [col[2] for col in columns]
        else:
            # Auto-calculate column widths based on content
            if table_data:
                col_widths = [1.5*inch] * len(table_data[0])
                for i in range(min(5, len(table_data))):  # Check first 5 rows
                    for j, cell in enumerate(table_data[i]):
                        width = len(str(cell)) * 7  # Approximate width in points
                        col_widths[j] = max(col_widths[j], min(width, 3*inch))
        
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Define alternating row colors
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F2F2F2'), colors.white]),
        ]
        
        table.setStyle(TableStyle(style_commands))
        return table
    
    @staticmethod
    def export_to_json(data: List[Dict[str, Any]], filename: str = None, 
                      indent: int = 2, sort_keys: bool = True,
                      datetime_format: str = 'iso') -> Union[str, str]:
        """
        Export data to JSON format
        
        Args:
            data: Data to export
            filename: Output filename (optional)
            indent: JSON indentation level
            sort_keys: Sort JSON keys alphabetically
            datetime_format: 'iso' or 'string'
            
        Returns:
            JSON string or filename
        """
        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                if datetime_format == 'iso':
                    return obj.isoformat()
                else:
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Type {type(obj)} not serializable")
        
        json_data = json.dumps(data, indent=indent, sort_keys=sort_keys, 
                              default=json_serializer, ensure_ascii=False)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_data)
            return filename
        
        return json_data
    
    @staticmethod
    def export_to_text(data: List[Dict[str, Any]], filename: str = None, 
                      delimiter: str = '\t', include_headers: bool = True,
                      max_column_width: int = 50) -> Union[str, str]:
        """
        Export data to plain text format
        
        Args:
            data: List of dictionaries
            filename: Output filename (optional)
            delimiter: Column delimiter
            include_headers: Whether to include headers
            max_column_width: Maximum column width for text wrapping
            
        Returns:
            Text string or filename
        """
        if not data:
            return "" if filename is None else filename
        
        lines = []
        
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            
            if include_headers:
                lines.append(delimiter.join(headers))
                lines.append(delimiter.join(['-' * len(h) for h in headers]))
            
            for row in data:
                row_values = []
                for header in headers:
                    value = row.get(header, '')
                    if isinstance(value, (date, datetime)):
                        value = value.strftime('%Y-%m-%d %H:%M')
                    elif value is None:
                        value = ''
                    else:
                        value = str(value)
                    
                    # Truncate long values
                    if len(value) > max_column_width:
                        value = value[:max_column_width-3] + '...'
                    
                    row_values.append(value)
                lines.append(delimiter.join(row_values))
        else:
            # Assume list of lists
            for row in data:
                lines.append(delimiter.join(str(item) for item in row))
        
        text_data = '\n'.join(lines)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_data)
            return filename
        
        return text_data
    
    @staticmethod
    def export_tasks_summary(tasks: List[Dict[str, Any]], filename: str = None, 
                            format: str = 'pdf', include_charts: bool = False) -> Any:
        """
        Export a comprehensive summary report of tasks
        
        Args:
            tasks: List of task dictionaries
            filename: Output filename
            format: Export format ('pdf', 'csv', 'excel', 'json', 'text')
            include_charts: Include charts in Excel export (if available)
            
        Returns:
            Exported data or filename
        """
        if not tasks:
            raise ValueError("No tasks to export")
        
        # Prepare summary data
        summary_data = []
        for task in tasks:
            summary_data.append({
                'Task ID': task.get('task_id', task.get('id', '')),
                'Title': task.get('title', ''),
                'Description': task.get('description', '')[:100] + '...' if task.get('description') and len(task.get('description')) > 100 else task.get('description', ''),
                'Priority': task.get('priority', ''),
                'Status': task.get('status', ''),
                'Category': task.get('category', ''),
                'Due Date': task.get('due_date', ''),
                'Assigned To': task.get('assigned_to', ''),
                'Created': task.get('created_at', ''),
                'Last Updated': task.get('updated_at', ''),
                'Completed': task.get('completed_at', ''),
                'Estimated Hours': task.get('estimated_hours', 0),
                'Actual Hours': task.get('actual_hours', 0),
                'Tags': ', '.join(task.get('tags', [])) if isinstance(task.get('tags'), list) else task.get('tags', '')
            })
        
        # Export based on format
        format = format.lower()
        
        if format == 'csv':
            return TaskExporter.export_to_csv(summary_data, filename)
        
        elif format == 'excel':
            df = pd.DataFrame(summary_data)
            
            if filename:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Tasks Summary', index=False)
                    
                    # Add summary statistics sheet
                    summary_stats = TaskExporter._calculate_task_statistics(tasks)
                    stats_df = pd.DataFrame([summary_stats])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                    
                    # Add charts if requested and matplotlib is available
                    if include_charts:
                        try:
                            import matplotlib.pyplot as plt
                            from openpyxl.drawing.image import Image
                            from io import BytesIO
                            
                            # Create status distribution chart
                            status_counts = {}
                            for task in tasks:
                                status = task.get('status', 'Unknown')
                                status_counts[status] = status_counts.get(status, 0) + 1
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.bar(status_counts.keys(), status_counts.values())
                            ax.set_title('Task Status Distribution')
                            ax.set_xlabel('Status')
                            ax.set_ylabel('Count')
                            plt.tight_layout()
                            
                            # Save chart to buffer
                            chart_buffer = BytesIO()
                            plt.savefig(chart_buffer, format='png')
                            plt.close()
                            
                            # Add chart to Excel
                            chart_buffer.seek(0)
                            img = Image(chart_buffer)
                            img.width = 400
                            img.height = 300
                            writer.sheets['Statistics'].add_image(img, 'A10')
                            
                        except ImportError:
                            pass  # matplotlib not available
                
                return filename
            
            else:
                return TaskExporter.export_to_excel(summary_data, sheet_name='Tasks Summary')
        
        elif format == 'pdf':
            columns = [
                ('Task ID', 'Task ID', 0.8*inch),
                ('Title', 'Title', 1.5*inch),
                ('Priority', 'Priority', 0.6*inch),
                ('Status', 'Status', 0.8*inch),
                ('Category', 'Category', 0.8*inch),
                ('Due Date', 'Due Date', 1.0*inch),
                ('Assigned To', 'Assigned To', 1.0*inch)
            ]
            return TaskExporter.export_to_pdf(
                summary_data,
                title="Tasks Summary Report",
                columns=columns,
                filename=filename,
                include_summary=True
            )
        
        elif format == 'json':
            return TaskExporter.export_to_json(summary_data, filename)
        
        elif format == 'text':
            return TaskExporter.export_to_text(summary_data, filename)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _calculate_task_statistics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate task statistics"""
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.get('status') == 'Completed')
        overdue_tasks = sum(1 for t in tasks if t.get('due_date') and 
                           datetime.strptime(t['due_date'], '%Y-%m-%d').date() < date.today() 
                           and t.get('status') != 'Completed')
        
        priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        status_counts = {}
        category_counts = {}
        
        for task in tasks:
            priority = task.get('priority', 'Unknown')
            status = task.get('status', 'Unknown')
            category = task.get('category', 'Uncategorized')
            
            if priority in priority_counts:
                priority_counts[priority] += 1
            
            status_counts[status] = status_counts.get(status, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'Total Tasks': total_tasks,
            'Completed Tasks': completed_tasks,
            'Completion Rate': f"{(completed_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%",
            'Overdue Tasks': overdue_tasks,
            'High Priority Tasks': priority_counts['High'],
            'Medium Priority Tasks': priority_counts['Medium'],
            'Low Priority Tasks': priority_counts['Low'],
            'Most Common Status': max(status_counts.items(), key=lambda x: x[1])[0] if status_counts else 'N/A',
            'Most Common Category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else 'N/A'
        }
    
    @staticmethod
    def export_tasks_by_status(tasks: List[Dict[str, Any]], filename: str = None, 
                              format: str = 'excel') -> Any:
        """
        Export tasks grouped by status
        
        Args:
            tasks: List of task dictionaries
            filename: Output filename
            format: Export format
            
        Returns:
            Exported data or filename
        """
        # Group tasks by status
        tasks_by_status = {}
        for task in tasks:
            status = task.get('status', 'Unknown')
            if status not in tasks_by_status:
                tasks_by_status[status] = []
            tasks_by_status[status].append(task)
        
        if format.lower() == 'excel':
            if filename:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Write summary sheet
                    summary_data = []
                    for status, status_tasks in tasks_by_status.items():
                        summary_data.append({
                            'Status': status,
                            'Count': len(status_tasks),
                            'High Priority': len([t for t in status_tasks if t.get('priority') == 'High']),
                            'Medium Priority': len([t for t in status_tasks if t.get('priority') == 'Medium']),
                            'Low Priority': len([t for t in status_tasks if t.get('priority') == 'Low']),
                            'Overdue': len([t for t in status_tasks if t.get('due_date') and 
                                           datetime.strptime(t['due_date'], '%Y-%m-%d').date() < date.today()])
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Write sheets for each status
                    for status, status_tasks in tasks_by_status.items():
                        status_df = pd.DataFrame(status_tasks)
                        sheet_name = status[:31]  # Excel sheet name limit
                        status_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                return filename
            
            else:
                # Return as bytes
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    summary_data = []
                    for status, status_tasks in tasks_by_status.items():
                        summary_data.append({
                            'Status': status,
                            'Count': len(status_tasks)
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                return buffer.getvalue()
        
        else:
            # For other formats, export grouped data
            grouped_data = []
            for status, status_tasks in tasks_by_status.items():
                grouped_data.append({
                    'group': status,
                    'tasks': status_tasks,
                    'count': len(status_tasks)
                })
            
            return TaskExporter.export_to_json(grouped_data, filename) if format.lower() == 'json' \
                   else TaskExporter.export_tasks_summary(tasks, filename, format)
    
    @staticmethod
    def export_project_report(projects: List[Dict[str, Any]], tasks: List[Dict[str, Any]], 
                            filename: str = None, format: str = 'pdf') -> Any:
        """
        Export project report with task breakdown
        
        Args:
            projects: List of project dictionaries
            tasks: List of task dictionaries
            filename: Output filename
            format: Export format
            
        Returns:
            Exported data or filename
        """
        # Group tasks by project
        tasks_by_project = {}
        for task in tasks:
            project_id = task.get('project_id')
            if project_id:
                if project_id not in tasks_by_project:
                    tasks_by_project[project_id] = []
                tasks_by_project[project_id].append(task)
        
        report_data = []
        for project in projects:
            project_tasks = tasks_by_project.get(project.get('id'), [])
            completed_tasks = sum(1 for t in project_tasks if t.get('status') == 'Completed')
            
            report_data.append({
                'Project ID': project.get('id', ''),
                'Project Name': project.get('name', ''),
                'Description': project.get('description', ''),
                'Start Date': project.get('start_date', ''),
                'End Date': project.get('end_date', ''),
                'Total Tasks': len(project_tasks),
                'Completed Tasks': completed_tasks,
                'Completion %': f"{(completed_tasks/len(project_tasks)*100):.1f}%" if project_tasks else "0%",
                'Status': project.get('status', '')
            })
        
        if format.lower() == 'pdf':
            return TaskExporter.export_to_pdf(
                report_data,
                title="Project Report",
                filename=filename,
                include_summary=True
            )
        elif format.lower() == 'excel':
            return TaskExporter.export_to_excel(report_data, sheet_name='Projects', filename=filename)
        else:
            return TaskExporter.export_tasks_summary(report_data, filename, format)
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported export formats
        
        Returns:
            List of format strings
        """
        return ['csv', 'excel', 'pdf', 'json', 'text']
    
    @staticmethod
    def get_format_info(format_type: str) -> Dict[str, str]:
        """
        Get information about a specific export format
        
        Args:
            format_type: Format type string
            
        Returns:
            Dictionary with format information
        """
        info = {
            'csv': {
                'name': 'CSV',
                'extension': '.csv',
                'description': 'Comma-separated values, good for spreadsheet import',
                'mime_type': 'text/csv',
                'best_for': 'Data exchange, spreadsheet import'
            },
            'excel': {
                'name': 'Excel',
                'extension': '.xlsx',
                'description': 'Microsoft Excel format with multiple sheets and formatting',
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'best_for': 'Detailed reports, data analysis'
            },
            'pdf': {
                'name': 'PDF',
                'extension': '.pdf',
                'description': 'Portable Document Format for professional printing and sharing',
                'mime_type': 'application/pdf',
                'best_for': 'Formal reports, documentation, printing'
            },
            'json': {
                'name': 'JSON',
                'extension': '.json',
                'description': 'JavaScript Object Notation for structured data interchange',
                'mime_type': 'application/json',
                'best_for': 'APIs, data transfer, web applications'
            },
            'text': {
                'name': 'Text',
                'extension': '.txt',
                'description': 'Plain text format with configurable delimiter',
                'mime_type': 'text/plain',
                'best_for': 'Simple data export, log files'
            }
        }
        
        return info.get(format_type.lower(), {})


# Convenience functions for backward compatibility
def export_to_csv(data, filename=None, **kwargs):
    return TaskExporter.export_to_csv(data, filename, **kwargs)

def export_to_excel(data, filename=None, **kwargs):
    return TaskExporter.export_to_excel(data, filename, **kwargs)

def export_to_pdf(data, filename=None, **kwargs):
    return TaskExporter.export_to_pdf(data, filename, **kwargs)

def export_to_json(data, filename=None, **kwargs):
    return TaskExporter.export_to_json(data, filename, **kwargs)

def export_to_text(data, filename=None, **kwargs):
    return TaskExporter.export_to_text(data, filename, **kwargs)

def export_tasks_summary(tasks, filename=None, format='pdf', **kwargs):
    return TaskExporter.export_tasks_summary(tasks, filename, format, **kwargs)

def export_tasks_by_status(tasks, filename=None, format='excel', **kwargs):
    return TaskExporter.export_tasks_by_status(tasks, filename, format, **kwargs)

def get_supported_formats():
    return TaskExporter.get_supported_formats()

def get_format_info(format_type):
    return TaskExporter.get_format_info(format_type)


if __name__ == "__main__":
    # Test the exporter with sample data
    sample_tasks = [
        {
            "id": 1,
            "title": "Fix login issue",
            "description": "Users unable to login with correct credentials",
            "priority": "High",
            "status": "In Progress",
            "category": "Bug",
            "due_date": "2024-12-15",
            "assigned_to": "John Doe",
            "created_at": "2024-11-01 09:00:00",
            "updated_at": "2024-11-10 14:30:00",
            "completed_at": None,
            "estimated_hours": 8,
            "actual_hours": 4,
            "tags": ["authentication", "bug", "urgent"]
        },
        {
            "id": 2,
            "title": "Update documentation",
            "description": "Update API documentation for new endpoints",
            "priority": "Medium",
            "status": "Completed",
            "category": "Documentation",
            "due_date": "2024-11-30",
            "assigned_to": "Jane Smith",
            "created_at": "2024-10-15 10:00:00",
            "updated_at": "2024-11-05 16:45:00",
            "completed_at": "2024-11-05 16:45:00",
            "estimated_hours": 16,
            "actual_hours": 12,
            "tags": ["documentation", "api"]
        },
        {
            "id": 3,
            "title": "Performance optimization",
            "description": "Optimize database queries for better performance",
            "priority": "High",
            "status": "Pending",
            "category": "Improvement",
            "due_date": "2024-12-20",
            "assigned_to": "Bob Wilson",
            "created_at": "2024-11-05 11:30:00",
            "updated_at": "2024-11-05 11:30:00",
            "completed_at": None,
            "estimated_hours": 24,
            "actual_hours": 0,
            "tags": ["performance", "database"]
        }
    ]
    
    print("Testing Task Exporter...")
    
    # Test CSV export
    csv_result = TaskExporter.export_to_csv(sample_tasks)
    print(f"CSV export: {len(csv_result)} characters")
    
    # Test JSON export
    json_result = TaskExporter.export_to_json(sample_tasks)
    print(f"JSON export: {len(json_result)} characters")
    
    # Test text export
    text_result = TaskExporter.export_to_text(sample_tasks)
    print(f"Text export: {len(text_result)} characters")
    
    # Test supported formats
    formats = TaskExporter.get_supported_formats()
    print(f"Supported formats: {formats}")
    
    # Test format info
    pdf_info = TaskExporter.get_format_info('pdf')
    print(f"PDF format info: {pdf_info}")
    
    print("\nExporter test completed successfully!")