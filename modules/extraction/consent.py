"""
EXTRACTION CONSENT MODULE - Consent-Based Extraction Control
Manages consent validation, module access control, and extraction permissions

This module provides:
- Consent level validation for extraction
- Module-specific consent requirements
- Media viewer feature consent checks
- Extraction permission checking
- Consent-based feature access control
- Audit trail for consent decisions
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from modules.consent.models import ConsentLevel, get_consent_manager

logger = logging.getLogger(__name__)

# ============================================================================
# MODULE CONSENT REQUIREMENTS
# ============================================================================

MODULE_MIN_LEVELS = {
    'device_info': ConsentLevel.STANDARD,
    'communications': ConsentLevel.LEGAL,
    'location': ConsentLevel.STANDARD,
    'security': ConsentLevel.FULL,
    'media': ConsentLevel.FULL,
    'system': ConsentLevel.FULL
}

# ============================================================================
# MEDIA VIEWER FEATURES & CONSENT REQUIREMENTS
# ============================================================================

MEDIA_VIEWER_FEATURES = {
    'corruption_detection': {
        'name': 'Corruption Detection',
        'description': 'Detect file corruption and integrity issues',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Detection & Scanning',
        'enabled': True
    },
    'file_recovery_scan': {
        'name': 'File Recovery Scan',
        'description': 'Scan for recoverable files using signature detection',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Detection & Scanning',
        'enabled': True
    },
    'ai_image_recovery': {
        'name': 'AI Image Recovery',
        'description': 'AI-powered image reconstruction using pattern recognition',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'AI-Powered Recovery',
        'enabled': True
    },
    'ai_video_recovery': {
        'name': 'AI Video Recovery',
        'description': 'AI-powered video frame recovery using temporal analysis',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'AI-Powered Recovery',
        'enabled': True
    },
    'smart_file_recovery': {
        'name': 'Smart File Recovery',
        'description': 'Smart file recovery using multiple AI techniques',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'AI-Powered Recovery',
        'enabled': True
    },
    'performance_optimization': {
        'name': 'Performance Optimization',
        'description': 'Optimize recovery performance with caching and parallelization',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Performance & Analysis',
        'enabled': True
    },
    'file_integrity_check': {
        'name': 'File Integrity Check',
        'description': 'Compare integrity between original and recovered files',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Performance & Analysis',
        'enabled': True
    },
    'quality_assessment': {
        'name': 'Quality Assessment',
        'description': 'Predictive analysis for recovery success and quality rating',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Performance & Analysis',
        'enabled': True
    },
    'recovery_report': {
        'name': 'Recovery Report',
        'description': 'Generate comprehensive recovery performance report',
        'required_consent': ConsentLevel.FULL,
        'module': 'media',
        'category': 'Performance & Analysis',
        'enabled': True
    }
}

# ============================================================================
# EXTRACTION CONSENT MANAGER
# ============================================================================

class ExtractionConsentManager:
    """Manage consent for extraction operations"""
    
    def __init__(self):
        self.consent_manager = get_consent_manager()
        self.audit_trail: List[Dict[str, Any]] = []
    
    def check_module_consent(self, current_level: ConsentLevel, module_name: str) -> Tuple[bool, str]:
        """Check if current consent level allows module extraction"""
        if module_name not in MODULE_MIN_LEVELS:
            msg = f"Unknown module: {module_name}"
            logger.error(f"❌ {msg}")
            return False, msg
        
        min_level = MODULE_MIN_LEVELS[module_name]
        if current_level.value >= min_level.value:
            msg = f"Consent level {current_level.name} allows {module_name} extraction"
            logger.info(f"✅ Consent check PASSED for {module_name}")
            return True, msg
        else:
            msg = f"Insufficient consent for {module_name}. Required: {min_level.name}, Current: {current_level.name}"
            logger.warning(f"❌ Consent check FAILED for {module_name}")
            return False, msg
    
    def check_media_viewer_feature(self, current_level: ConsentLevel, feature_name: str) -> Tuple[bool, str]:
        """Check if current consent level allows media viewer feature"""
        if feature_name not in MEDIA_VIEWER_FEATURES:
            msg = f"Unknown media viewer feature: {feature_name}"
            logger.error(f"❌ {msg}")
            return False, msg
        
        feature = MEDIA_VIEWER_FEATURES[feature_name]
        required_level = feature['required_consent']
        
        if current_level.value >= required_level.value:
            msg = f"Consent level {current_level.name} allows {feature['name']}"
            logger.info(f"✅ Media feature ALLOWED: {feature_name}")
            return True, msg
        else:
            msg = f"Insufficient consent for {feature['name']}. Required: {required_level.name}, Current: {current_level.name}"
            logger.warning(f"❌ Media feature BLOCKED: {feature_name}")
            return False, msg
    
    def check_extraction_consent(self, case_id: str, modules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check consent for extraction with specific modules"""
        try:
            session = self.consent_manager.get_session(case_id)
            
            if not session or not session.level:
                result = {
                    'allowed': False,
                    'consent_level': None,
                    'modules_allowed': [],
                    'modules_blocked': list(MODULE_MIN_LEVELS.keys()),
                    'message': f"No consent found for case {case_id}",
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ No consent session for case: {case_id}")
                return result
            
            modules_to_check = modules or list(MODULE_MIN_LEVELS.keys())
            modules_allowed = []
            modules_blocked = []
            
            for module in modules_to_check:
                allowed, msg = self.check_module_consent(session.level, module)
                if allowed:
                    modules_allowed.append(module)
                else:
                    modules_blocked.append(module)
            
            overall_allowed = len(modules_allowed) > 0
            
            result = {
                'allowed': overall_allowed,
                'consent_level': session.level.name,
                'consent_value': session.level.value,
                'modules_allowed': modules_allowed,
                'modules_blocked': modules_blocked,
                'message': f"Extraction allowed for {len(modules_allowed)} modules, blocked for {len(modules_blocked)}",
                'timestamp': datetime.now().isoformat()
            }
            
            return result
                    
        except Exception as e:
            logger.error(f"❌ Error checking extraction consent: {e}")
            return {
                'allowed': False,
                'consent_level': None,
                'modules_allowed': [],
                'modules_blocked': list(MODULE_MIN_LEVELS.keys()),
                'message': f"Error checking consent: {str(e)}",
                'error': str(e)
            }
    
    def check_media_viewer_features(self, case_id: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check consent for media viewer features"""
        try:
            session = self.consent_manager.get_session(case_id)
            
            if not session or not session.level:
                result = {
                    'allowed': False,
                    'consent_level': None,
                    'features_allowed': [],
                    'features_blocked': list(MEDIA_VIEWER_FEATURES.keys()),
                    'message': f"No consent found for case {case_id}",
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ No consent session for case: {case_id}")
                return result
            
            features_to_check = features or list(MEDIA_VIEWER_FEATURES.keys())
            features_allowed = []
            features_blocked = []
            
            for feature in features_to_check:
                allowed, msg = self.check_media_viewer_feature(session.level, feature)
                if allowed:
                    features_allowed.append(feature)
                else:
                    features_blocked.append(feature)
            
            overall_allowed = len(features_allowed) > 0
            
            result = {
                'allowed': overall_allowed,
                'consent_level': session.level.name,
                'consent_value': session.level.value,
                'features_allowed': features_allowed,
                'features_blocked': features_blocked,
                'message': f"Media features allowed: {len(features_allowed)}, blocked: {len(features_blocked)}",
                'timestamp': datetime.now().isoformat()
            }
            
            return result
                    
        except Exception as e:
            logger.error(f"❌ Error checking media viewer features: {e}")
            return {
                'allowed': False,
                'consent_level': None,
                'features_allowed': [],
                'features_blocked': list(MEDIA_VIEWER_FEATURES.keys()),
                'message': f"Error checking consent: {str(e)}",
                'error': str(e)
            }
    
    def get_module_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get all module requirements"""
        requirements = {}
        for module, min_level in MODULE_MIN_LEVELS.items():
            requirements[module] = {
                'module': module,
                'required_level': min_level.name,
                'required_value': min_level.value
            }
        return requirements
    
    def get_media_viewer_features_info(self) -> Dict[str, Dict[str, Any]]:
        """Get all media viewer features information"""
        return MEDIA_VIEWER_FEATURES

# ============================================================================
# EXTRACTION CONSENT VALIDATOR
# ============================================================================

class ExtractionConsentValidator:
    """Validate extraction requests against consent"""
    
    def __init__(self):
        self.consent_manager = ExtractionConsentManager()
    
    def validate_extraction_request(
        self,
        case_id: str,
        device_id: str,
        modules: Optional[List[str]] = None,
        dev_mode: bool = False
    ) -> Dict[str, Any]:
        """Validate extraction request"""
        errors = []
        
        if not case_id:
            errors.append("Case ID is required")
        if not device_id:
            errors.append("Device ID is required")
        
        if errors:
            return {
                'valid': False,
                'case_id': case_id,
                'device_id': device_id,
                'modules': modules or [],
                'consent_level': None,
                'message': "Validation failed",
                'errors': errors
            }
        
        consent_result = self.consent_manager.check_extraction_consent(case_id, modules)
        
        if dev_mode:
            logger.warning("⚠️ Dev mode: Consent checks bypassed")
            return {
                'valid': True,
                'case_id': case_id,
                'device_id': device_id,
                'modules': modules or list(MODULE_MIN_LEVELS.keys()),
                'consent_level': consent_result.get('consent_level', 'DEV_MODE'),
                'message': "Dev mode: Extraction allowed (consent bypassed)",
                'dev_mode': True,
                'errors': []
            }
        
        if not consent_result['allowed']:
            errors.extend([
                f"Insufficient consent for modules: {', '.join(consent_result['modules_blocked'])}"
            ])
        
        return {
            'valid': consent_result['allowed'],
            'case_id': case_id,
            'device_id': device_id,
            'modules': consent_result['modules_allowed'],
            'blocked_modules': consent_result['modules_blocked'],
            'consent_level': consent_result.get('consent_level'),
            'message': consent_result['message'],
            'errors': errors
        }

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_extraction_consent_manager: Optional[ExtractionConsentManager] = None
_extraction_consent_validator: Optional[ExtractionConsentValidator] = None

def get_extraction_consent_manager() -> ExtractionConsentManager:
    """Get global extraction consent manager"""
    global _extraction_consent_manager
    if _extraction_consent_manager is None:
        _extraction_consent_manager = ExtractionConsentManager()
    return _extraction_consent_manager

def get_extraction_consent_validator() -> ExtractionConsentValidator:
    """Get global extraction consent validator"""
    global _extraction_consent_validator
    if _extraction_consent_validator is None:
        _extraction_consent_validator = ExtractionConsentValidator()
    return _extraction_consent_validator
