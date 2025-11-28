"""
ENHANCED REPORT GENERATOR - Integrates with Database and API

Provides:
- Database integration for report storage
- API integration for external data
- Intelligence integration for analysis
- Advanced report generation
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from modules.shared.database import DatabaseManager
from modules.shared.api import APIClient

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED REPORT GENERATOR CLASS
# ============================================================================

class EnhancedReportGenerator:
    """Enhanced report generator with database and API integration"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api = APIClient()
        self.reports = {}
        self.report_history = []
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def initialize(self) -> bool:
        """Initialize enhanced report generator"""
        try:
            # Connect to database
            if not self.db.connect():
                logger.error("Failed to connect to database")
                return False
            
            # Initialize API
            self._initialize_api()
            
            logger.info("Enhanced report generator initialized")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _initialize_api(self) -> None:
        """Initialize API endpoints"""
        self.api.register_endpoint(
            'generate_report',
            'POST',
            'reports/generate',
            'Generate forensic report'
        )
        self.api.register_endpoint(
            'get_report',
            'GET',
            'reports/get',
            'Get generated report'
        )
        self.api.register_endpoint(
            'export_report',
            'POST',
            'reports/export',
            'Export report'
        )
    
    # ========================================================================
    # REPORT GENERATION WITH DATABASE
    # ========================================================================
    
    def generate_report(self, case_id: str, report_type: str, 
                       extraction_data: Dict[str, Any],
                       analysis_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive report with database storage"""
        try:
            logger.info(f"Generating {report_type} report for case {case_id}")
            
            # Create report record in database
            report_record = self.db.create('reports', {
                'case_id': case_id,
                'report_type': report_type,
                'status': 'generating',
                'extraction_data': extraction_data,
                'analysis_data': analysis_data or {}
            })
            
            # Generate report content
            report_content = self._generate_report_content(
                case_id, report_type, extraction_data, analysis_data
            )
            
            # Update report record
            self.db.update('reports', report_record['id'], {
                'content': report_content,
                'status': 'completed',
                'generated_at': datetime.now().isoformat()
            })
            
            # Store in memory
            self.reports[report_record['id']] = report_content
            
            # Log report generation
            history_record = {
                'case_id': case_id,
                'report_type': report_type,
                'report_id': report_record['id'],
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            self.report_history.append(history_record)
            
            logger.info(f"Report generated successfully: {report_record['id']}")
            
            return {
                'success': True,
                'report_id': report_record['id'],
                'case_id': case_id,
                'report_type': report_type,
                'content': report_content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_report_content(self, case_id: str, report_type: str,
                                extraction_data: Dict[str, Any],
                                analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report content"""
        report = {
            'case_id': case_id,
            'report_type': report_type,
            'generated_date': datetime.now().isoformat(),
            'sections': []
        }
        
        # Executive Summary
        report['sections'].append({
            'title': 'Executive Summary',
            'content': f"Forensic analysis of case {case_id}",
            'type': 'summary'
        })
        
        # Extraction Summary
        if extraction_data:
            report['sections'].append({
                'title': 'Extraction Summary',
                'content': f"Extracted {len(extraction_data)} data items",
                'data': extraction_data,
                'type': 'extraction'
            })
        
        # Analysis Results
        if analysis_data:
            report['sections'].append({
                'title': 'Analysis Results',
                'content': 'Detailed analysis findings',
                'data': analysis_data,
                'type': 'analysis'
            })
        
        # Recommendations
        report['sections'].append({
            'title': 'Recommendations',
            'content': 'Further investigation recommended',
            'type': 'recommendations'
        })
        
        return report
    
    # ========================================================================
    # REPORT RETRIEVAL WITH DATABASE
    # ========================================================================
    
    def get_report(self, report_id: int) -> Dict[str, Any]:
        """Get report from database"""
        try:
            reports = self.db.read('reports', report_id)
            
            if reports:
                return {
                    'success': True,
                    'report': reports[0],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': 'Report not found'}
        except Exception as e:
            logger.error(f"Get report failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_case_reports(self, case_id: str) -> Dict[str, Any]:
        """Get all reports for a case"""
        try:
            reports = self.db.query('reports', {'case_id': case_id})
            
            return {
                'success': True,
                'case_id': case_id,
                'reports': reports,
                'count': len(reports),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Get case reports failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # REPORT EXPORT WITH API
    # ========================================================================
    
    def export_report(self, report_id: int, format_type: str = 'pdf') -> Dict[str, Any]:
        """Export report via API"""
        try:
            # Get report from database
            report_data = self.get_report(report_id)
            
            if not report_data['success']:
                return report_data
            
            # Call API to export
            response = self.api.post('reports/export', {
                'report_id': report_id,
                'format': format_type,
                'data': report_data['report']
            })
            
            if response['success']:
                logger.info(f"Report {report_id} exported as {format_type}")
                return {
                    'success': True,
                    'report_id': report_id,
                    'format': format_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': 'Export failed'}
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # REPORT MANAGEMENT
    # ========================================================================
    
    def archive_report(self, report_id: int) -> bool:
        """Archive report"""
        try:
            return self.db.update('reports', report_id, {
                'status': 'archived',
                'archived_at': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Archive failed: {e}")
            return False
    
    def delete_report(self, report_id: int) -> bool:
        """Delete report"""
        try:
            return self.db.delete('reports', report_id)
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """Get report statistics"""
        try:
            all_reports = self.db.read('reports')
            
            stats = {
                'total_reports': len(all_reports),
                'by_type': {},
                'by_status': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for report in all_reports:
                report_type = report.get('report_type', 'unknown')
                status = report.get('status', 'unknown')
                
                stats['by_type'][report_type] = stats['by_type'].get(report_type, 0) + 1
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            return stats
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}
    
    def get_report_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get report generation history"""
        return self.report_history[-limit:]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_statistics()
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API statistics"""
        return self.api.get_statistics()

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_enhanced_report_generator() -> EnhancedReportGenerator:
    """Factory function to create enhanced report generator"""
    return EnhancedReportGenerator()
