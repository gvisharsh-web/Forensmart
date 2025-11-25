"""
ForenSmart Error Checker & Validator
====================================

Comprehensive error checking and validation:
- Storage integrity checks
- Artifact consistency validation
- Orphaned file detection
- Permission verification
- Corruption detection
- Detailed error reporting

Author: ForenSmart Development Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorChecker:
    """Check for errors and inconsistencies in ForenSmart."""
    
    @staticmethod
    def check_storage_integrity() -> Dict[str, Any]:
        """Check overall storage integrity."""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'errors': [],
            'warnings': [],
            'info': [],
            'checks': {}
        }
        
        # Check artifacts directory
        artifacts_check = ErrorChecker._check_artifacts_directory()
        results['checks']['artifacts'] = artifacts_check
        if artifacts_check['errors']:
            results['errors'].extend(artifacts_check['errors'])
        if artifacts_check['warnings']:
            results['warnings'].extend(artifacts_check['warnings'])
        
        # Check reports directory
        reports_check = ErrorChecker._check_reports_directory()
        results['checks']['reports'] = reports_check
        if reports_check['errors']:
            results['errors'].extend(reports_check['errors'])
        if reports_check['warnings']:
            results['warnings'].extend(reports_check['warnings'])
        
        # Check consent records
        consent_check = ErrorChecker._check_consent_records()
        results['checks']['consent'] = consent_check
        if consent_check['errors']:
            results['errors'].extend(consent_check['errors'])
        if consent_check['warnings']:
            results['warnings'].extend(consent_check['warnings'])
        
        # Check for orphaned files
        orphaned_check = ErrorChecker._check_orphaned_files()
        results['checks']['orphaned'] = orphaned_check
        if orphaned_check['errors']:
            results['errors'].extend(orphaned_check['errors'])
        if orphaned_check['warnings']:
            results['warnings'].extend(orphaned_check['warnings'])
        
        # Check permissions
        permissions_check = ErrorChecker._check_permissions()
        results['checks']['permissions'] = permissions_check
        if permissions_check['errors']:
            results['errors'].extend(permissions_check['errors'])
        if permissions_check['warnings']:
            results['warnings'].extend(permissions_check['warnings'])
        
        # Determine overall status
        if results['errors']:
            results['status'] = 'error'
        elif results['warnings']:
            results['status'] = 'warning'
        else:
            results['status'] = 'ok'
        
        return results
    
    @staticmethod
    def _check_artifacts_directory() -> Dict[str, Any]:
        """Check artifacts directory integrity."""
        
        check = {
            'name': 'Artifacts Directory',
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        artifacts_dir = 'artifacts'
        
        if not os.path.exists(artifacts_dir):
            check['warnings'].append(f"Artifacts directory does not exist: {artifacts_dir}")
            return check
        
        if not os.path.isdir(artifacts_dir):
            check['errors'].append(f"Artifacts path is not a directory: {artifacts_dir}")
            return check
        
        try:
            # Count cases and files
            case_count = 0
            total_files = 0
            total_size = 0
            
            for case_dir in os.listdir(artifacts_dir):
                case_path = os.path.join(artifacts_dir, case_dir)
                
                if not os.path.isdir(case_path):
                    check['warnings'].append(f"Non-directory item in artifacts: {case_dir}")
                    continue
                
                case_count += 1
                
                # Count files in case
                for root, dirs, files in os.walk(case_path):
                    total_files += len(files)
                    for file in files:
                        try:
                            total_size += os.path.getsize(os.path.join(root, file))
                        except Exception as e:
                            check['warnings'].append(f"Cannot get size of {file}: {e}")
            
            check['stats'] = {
                'case_count': case_count,
                'total_files': total_files,
                'total_size': total_size
            }
            
        except Exception as e:
            check['errors'].append(f"Error checking artifacts directory: {str(e)}")
        
        return check
    
    @staticmethod
    def _check_reports_directory() -> Dict[str, Any]:
        """Check reports directory integrity."""
        
        check = {
            'name': 'Reports Directory',
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        reports_dir = 'reports'
        
        if not os.path.exists(reports_dir):
            check['warnings'].append(f"Reports directory does not exist: {reports_dir}")
            return check
        
        try:
            case_count = 0
            total_files = 0
            invalid_json = []
            
            for case_dir in os.listdir(reports_dir):
                case_path = os.path.join(reports_dir, case_dir)
                
                if not os.path.isdir(case_path):
                    continue
                
                case_count += 1
                
                # Check for results.json
                results_file = os.path.join(case_path, 'results.json')
                if os.path.exists(results_file):
                    total_files += 1
                    
                    # Validate JSON
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        invalid_json.append(f"{case_dir}/results.json: {str(e)}")
                    except Exception as e:
                        check['warnings'].append(f"Cannot read {results_file}: {e}")
            
            if invalid_json:
                check['errors'].extend([f"Invalid JSON: {item}" for item in invalid_json])
            
            check['stats'] = {
                'case_count': case_count,
                'total_files': total_files
            }
            
        except Exception as e:
            check['errors'].append(f"Error checking reports directory: {str(e)}")
        
        return check
    
    @staticmethod
    def _check_consent_records() -> Dict[str, Any]:
        """Check consent records integrity."""
        
        check = {
            'name': 'Consent Records',
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        consent_dir = 'consent_records'
        
        if not os.path.exists(consent_dir):
            check['warnings'].append(f"Consent records directory does not exist: {consent_dir}")
            return check
        
        try:
            case_count = 0
            total_files = 0
            
            for case_dir in os.listdir(consent_dir):
                case_path = os.path.join(consent_dir, case_dir)
                
                if not os.path.isdir(case_path):
                    continue
                
                case_count += 1
                
                # Count files
                for root, dirs, files in os.walk(case_path):
                    total_files += len(files)
            
            check['stats'] = {
                'case_count': case_count,
                'total_files': total_files
            }
            
        except Exception as e:
            check['errors'].append(f"Error checking consent records: {str(e)}")
        
        return check
    
    @staticmethod
    def _check_orphaned_files() -> Dict[str, Any]:
        """Check for orphaned files."""
        
        check = {
            'name': 'Orphaned Files',
            'errors': [],
            'warnings': [],
            'orphaned_cases': []
        }
        
        try:
            if not os.path.exists('artifacts'):
                return check
            
            # Get list of cases with consent records
            consent_cases = set()
            if os.path.exists('consent_records'):
                consent_cases = set(os.listdir('consent_records'))
            
            # Check each artifact case
            for case_dir in os.listdir('artifacts'):
                case_path = os.path.join('artifacts', case_dir)
                
                if not os.path.isdir(case_path):
                    continue
                
                if case_dir not in consent_cases:
                    check['orphaned_cases'].append(case_dir)
                    check['warnings'].append(f"Orphaned case found: {case_dir}")
            
        except Exception as e:
            check['errors'].append(f"Error checking for orphaned files: {str(e)}")
        
        return check
    
    @staticmethod
    def _check_permissions() -> Dict[str, Any]:
        """Check file permissions and write access."""
        
        check = {
            'name': 'Permissions',
            'errors': [],
            'warnings': [],
            'writable_dirs': {}
        }
        
        dirs_to_check = [
            'artifacts',
            'reports',
            'consent_records',
            'audit',
            'case_snapshots'
        ]
        
        for dir_name in dirs_to_check:
            if os.path.exists(dir_name):
                is_writable = os.access(dir_name, os.W_OK)
                check['writable_dirs'][dir_name] = is_writable
                
                if not is_writable:
                    check['errors'].append(f"Directory not writable: {dir_name}")
            else:
                # Try to create it
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    check['writable_dirs'][dir_name] = True
                except Exception as e:
                    check['errors'].append(f"Cannot create directory {dir_name}: {e}")
        
        return check
    
    @staticmethod
    def validate_case_consistency(case_id: str) -> Dict[str, Any]:
        """Validate consistency of a specific case."""
        
        results = {
            'case_id': case_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check artifacts exist
        artifacts_path = os.path.join('artifacts', case_id)
        if not os.path.exists(artifacts_path):
            results['warnings'].append(f"No artifacts found for case {case_id}")
        else:
            results['checks']['artifacts_exist'] = True
        
        # Check reports exist
        reports_path = os.path.join('reports', case_id)
        if not os.path.exists(reports_path):
            results['warnings'].append(f"No reports found for case {case_id}")
        else:
            results['checks']['reports_exist'] = True
            
            # Validate results.json
            results_file = os.path.join(reports_path, 'results.json')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    results['checks']['results_json_valid'] = True
                except json.JSONDecodeError as e:
                    results['errors'].append(f"Invalid results.json: {str(e)}")
                    results['checks']['results_json_valid'] = False
        
        # Check consent records exist
        consent_path = os.path.join('consent_records', case_id)
        if not os.path.exists(consent_path):
            results['warnings'].append(f"No consent records found for case {case_id}")
        else:
            results['checks']['consent_exist'] = True
        
        # Determine status
        if results['errors']:
            results['status'] = 'error'
        elif results['warnings']:
            results['status'] = 'warning'
        
        return results
    
    @staticmethod
    def get_error_report() -> str:
        """Generate comprehensive error report."""
        
        report = "=" * 80 + "\n"
        report += "ForenSmart Error & Integrity Report\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += "=" * 80 + "\n\n"
        
        # Run all checks
        integrity = ErrorChecker.check_storage_integrity()
        
        # Summary
        report += "SUMMARY\n"
        report += "-" * 80 + "\n"
        report += f"Status: {integrity['status'].upper()}\n"
        report += f"Errors: {len(integrity['errors'])}\n"
        report += f"Warnings: {len(integrity['warnings'])}\n\n"
        
        # Errors
        if integrity['errors']:
            report += "ERRORS\n"
            report += "-" * 80 + "\n"
            for error in integrity['errors']:
                report += f"❌ {error}\n"
            report += "\n"
        
        # Warnings
        if integrity['warnings']:
            report += "WARNINGS\n"
            report += "-" * 80 + "\n"
            for warning in integrity['warnings']:
                report += f"⚠️ {warning}\n"
            report += "\n"
        
        # Detailed checks
        report += "DETAILED CHECKS\n"
        report += "-" * 80 + "\n"
        for check_name, check_result in integrity['checks'].items():
            report += f"\n{check_name.upper()}\n"
            if check_result.get('stats'):
                for key, value in check_result['stats'].items():
                    report += f"  {key}: {value}\n"
        
        return report
