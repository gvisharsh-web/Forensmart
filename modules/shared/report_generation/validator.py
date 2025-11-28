"""
REPORT VALIDATOR - Validate Report Compliance

Provides functionality to validate report compliance with:
- IT Act of India requirements
- Evidence Act requirements
- Chain of custody requirements
- Data integrity requirements
"""

import logging
from typing import Dict, List, Any, Tuple

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# REPORT VALIDATOR CLASS
# ============================================================================

class ReportValidator:
    """
    Validate report compliance with legal and technical requirements.
    
    This class provides methods to validate that reports meet
    all necessary compliance standards.
    """
    
    def __init__(self):
        """Initialize report validator"""
        logger.info("ReportValidator initialized")
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    def validate_report_structure(self, report_content: str) -> Tuple[bool, List[str]]:
        """
        Validate report structure
        
        Args:
            report_content: Report content
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check if report is not empty
            if not report_content or len(report_content.strip()) == 0:
                errors.append("Report content is empty")
            
            # Check for required sections
            required_sections = [
                "CASE INFORMATION",
                "DEVICE INFORMATION",
                "EXTRACTION",
                "FINDINGS",
                "CONCLUSIONS"
            ]
            
            for section in required_sections:
                if section not in report_content:
                    errors.append(f"Missing required section: {section}")
            
            is_valid = len(errors) == 0
            logger.info(f"Report structure validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating report structure: {str(e)}")
            return False, [str(e)]
    
    def validate_data_integrity(self, report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data integrity
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check for required fields
            required_fields = ['case_id', 'case_details', 'extraction_results']
            
            for field in required_fields:
                if field not in report_data:
                    errors.append(f"Missing required field: {field}")
            
            # Check case details
            if 'case_details' in report_data:
                case_details = report_data['case_details']
                required_case_fields = ['investigator', 'device_type']
                
                for field in required_case_fields:
                    if field not in case_details:
                        errors.append(f"Missing case detail: {field}")
            
            # Check extraction results
            if 'extraction_results' in report_data:
                results = report_data['extraction_results']
                required_result_fields = ['total_size', 'file_count']
                
                for field in required_result_fields:
                    if field not in results:
                        errors.append(f"Missing extraction result: {field}")
            
            is_valid = len(errors) == 0
            logger.info(f"Data integrity validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating data integrity: {str(e)}")
            return False, [str(e)]
    
    def validate_it_act_compliance(self, report_content: str) -> Tuple[bool, List[str]]:
        """
        Validate IT Act of India compliance
        
        Args:
            report_content: Report content
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check for IT Act compliance sections
            compliance_sections = [
                "INVESTIGATOR",
                "CHAIN OF CUSTODY",
                "HASH VERIFICATION",
                "CERTIFICATION"
            ]
            
            for section in compliance_sections:
                if section not in report_content:
                    errors.append(f"Missing IT Act compliance section: {section}")
            
            # Check for digital signature/certification
            if "SIGNATURE" not in report_content and "CERTIFICATION" not in report_content:
                errors.append("Missing digital signature or certification")
            
            is_valid = len(errors) == 0
            logger.info(f"IT Act compliance validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating IT Act compliance: {str(e)}")
            return False, [str(e)]
    
    def validate_chain_of_custody(self, report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate chain of custody
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check for chain of custody information
            if 'chain_of_custody' not in report_data:
                errors.append("Chain of custody information missing")
            else:
                coc = report_data['chain_of_custody']
                
                # Check required CoC fields
                required_coc_fields = ['seized_by', 'seized_date', 'device_condition']
                
                for field in required_coc_fields:
                    if field not in coc:
                        errors.append(f"Missing chain of custody field: {field}")
            
            is_valid = len(errors) == 0
            logger.info(f"Chain of custody validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating chain of custody: {str(e)}")
            return False, [str(e)]
    
    def validate_hash_verification(self, report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate hash verification
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check for hash verification
            if 'hash_verification' not in report_data:
                errors.append("Hash verification information missing")
            else:
                hash_info = report_data['hash_verification']
                
                # Check required hash fields
                required_hash_fields = ['source_hash', 'destination_hash', 'match_status']
                
                for field in required_hash_fields:
                    if field not in hash_info:
                        errors.append(f"Missing hash verification field: {field}")
                
                # Check if hashes match
                if 'match_status' in hash_info:
                    if hash_info['match_status'] != 'MATCH':
                        errors.append("Hash verification failed: Hashes do not match")
            
            is_valid = len(errors) == 0
            logger.info(f"Hash verification validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating hash verification: {str(e)}")
            return False, [str(e)]
    
    def validate_signatures(self, report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate digital signatures
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Tuple: (is_valid, list of errors)
        """
        try:
            errors = []
            
            # Check for signatures
            if 'signatures' not in report_data:
                errors.append("Signature information missing")
            else:
                signatures = report_data['signatures']
                
                # Check for examiner signature
                if 'examiner_signature' not in signatures:
                    errors.append("Missing examiner signature")
                
                # Check for supervisor signature (if required)
                if 'supervisor_signature' not in signatures:
                    errors.append("Missing supervisor signature")
            
            is_valid = len(errors) == 0
            logger.info(f"Signature validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Error validating signatures: {str(e)}")
            return False, [str(e)]
    
    def validate_all(self, report_content: str, report_data: Dict[str, Any]) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all compliance requirements
        
        Args:
            report_content: Report content
            report_data: Report data dictionary
            
        Returns:
            Dict: Dictionary with validation results
        """
        try:
            results = {
                'structure': self.validate_report_structure(report_content),
                'data_integrity': self.validate_data_integrity(report_data),
                'it_act_compliance': self.validate_it_act_compliance(report_content),
                'chain_of_custody': self.validate_chain_of_custody(report_data),
                'hash_verification': self.validate_hash_verification(report_data),
                'signatures': self.validate_signatures(report_data)
            }
            
            # Overall validation
            all_valid = all(result[0] for result in results.values())
            results['overall'] = (all_valid, [])
            
            logger.info(f"Overall validation: {'PASSED' if all_valid else 'FAILED'}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in overall validation: {str(e)}")
            return {}
    
    def get_validation_report(self, validation_results: Dict[str, Tuple[bool, List[str]]]) -> str:
        """
        Generate validation report
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            str: Validation report
        """
        try:
            report = "VALIDATION REPORT\n"
            report += "=" * 79 + "\n\n"
            
            for check_name, (is_valid, errors) in validation_results.items():
                status = "✓ PASSED" if is_valid else "✗ FAILED"
                report += f"{check_name.upper()}: {status}\n"
                
                if errors:
                    for error in errors:
                        report += f"  - {error}\n"
                
                report += "\n"
            
            logger.debug("Validation report generated")
            return report
        
        except Exception as e:
            logger.error(f"Error generating validation report: {str(e)}")
            return ""
