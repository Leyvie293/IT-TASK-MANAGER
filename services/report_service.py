@app.route('/reports/generate', methods=['POST'])
@login_required
def generate_report():
    from services.report_service import ReportService
    
    report_type = request.form.get('report_type')
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    format_type = request.form.get('format', 'html')
    
    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        end_date = end_date.replace(hour=23, minute=59, second=59)
    else:
        # Default to last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        end_date = end_date.replace(hour=23, minute=59, second=59)
    
    report_service = ReportService()
    
    if report_type == 'sla':
        result = report_service.generate_sla_report(start_date, end_date, format=format_type)
        
        if format_type == 'pdf':
            return send_file(
                result,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'sla_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            )
        elif format_type == 'csv':
            return Response(
                result,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=sla_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
        elif format_type == 'excel':
            return send_file(
                result,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'sla_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )
        else:
            return render_template('reports/sla_report.html', metrics=result)
    
    return redirect(url_for('reports_dashboard'))