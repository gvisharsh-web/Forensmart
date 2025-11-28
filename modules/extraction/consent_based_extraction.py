"""
CONSENT-BASED EXTRACTION MODULE
Single integrated module for all extraction with consent verification

Features:
- Token verification (hash + signature)
- Consent checking
- All extractors in one place
- Orchestration
- Results aggregation
- Audit logging
"""

from typing import Dict, List, Tuple
from datetime import datetime
import hashlib
import hmac
import json


# ============================================================================
# CONSENT TOKEN VERIFICATION
# ============================================================================

class ConsentTokenVerifier:
    """Verify consent tokens"""
    
    def __init__(self, secret_key: str = "forensmart-secret-key"):
        self.secret_key = secret_key
    
    def verify_token(self, token: str) -> Tuple[bool, str, Dict]:
        """Verify token and return consent data"""
        try:
            # Remove header if present
            if token.startswith("FORENSMART_CONSENT_TOKEN"):
                token = token.split('\n')[1]
            
            # Decode from Base64
            import base64
            decoded = base64.b64decode(token)
            token_data = json.loads(decoded)
            
            # Extract components
            consent_data = token_data.get('data', {})
            received_hash = token_data.get('hash', '')
            received_signature = token_data.get('signature', '')
            
            # Verify hash
            data_json = json.dumps(consent_data, sort_keys=True)
            calculated_hash = hashlib.sha256(data_json.encode()).hexdigest()
            
            if calculated_hash != received_hash:
                return False, "Hash mismatch - data tampered!", None
            
            # Verify signature
            calculated_signature = hmac.new(
                self.secret_key.encode(),
                data_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if calculated_signature != received_signature:
                return False, "Signature mismatch - not authentic!", None
            
            # Check expiry
            try:
                expiry = datetime.fromisoformat(consent_data.get('expiry_date', ''))
                if datetime.now() > expiry:
                    return False, "Consent expired!", None
            except:
                pass
            
            # Verify required fields
            required_fields = ['case_id', 'consent_level', 'approved_by', 'modules_allowed']
            missing_fields = [f for f in required_fields if f not in consent_data]
            
            if missing_fields:
                return False, f"Missing fields: {', '.join(missing_fields)}", None
            
            return True, "Consent verified!", consent_data
        
        except Exception as e:
            return False, f"Verification error: {str(e)}", None


# ============================================================================
# BASE EXTRACTOR
# ============================================================================

class BaseExtractor:
    """Base class for all extractors"""
    
    def __init__(self, consent_data: Dict):
        self.consent_data = consent_data
        self.case_id = consent_data.get('case_id')
        self.consent_level = consent_data.get('consent_level')
        self.modules_allowed = consent_data.get('modules_allowed', [])
        self.modules_blocked = consent_data.get('modules_blocked', [])
        self.extraction_log = []
    
    def check_consent(self, module_name: str) -> Tuple[bool, str]:
        """Check if module extraction is allowed"""
        
        if module_name not in self.modules_allowed:
            reason = f"Module '{module_name}' not allowed by consent level '{self.consent_level}'"
            self.log_extraction(module_name, 'blocked', reason)
            return False, reason
        
        if module_name in self.modules_blocked:
            reason = f"Module '{module_name}' is blocked"
            self.log_extraction(module_name, 'blocked', reason)
            return False, reason
        
        return True, "Consent verified"
    
    def log_extraction(self, module_name: str, status: str, details: str = ""):
        """Log extraction attempt"""
        self.extraction_log.append({
            'timestamp': datetime.now().isoformat(),
            'case_id': self.case_id,
            'module': module_name,
            'status': status,
            'details': details,
            'consent_level': self.consent_level
        })
    
    def get_extraction_log(self) -> List[Dict]:
        """Get extraction log"""
        return self.extraction_log


# ============================================================================
# EXTRACTORS FOR EACH MODULE
# ============================================================================

class DeviceInfoExtractor(BaseExtractor):
    """Extract device information"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract device info"""
        
        is_allowed, reason = self.check_consent('device_info')
        
        if not is_allowed:
            return {
                'module': 'device_info',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('device_info', 'started')
        
        try:
            device_info = {
                'device_id': device_id,
                'model': 'Device Model',
                'os_version': 'OS Version',
                'android_version': '14.0',
                'build_number': 'Build Number',
                'serial_number': 'Serial Number',
                'imei': 'IMEI Number',
                'phone_number': 'Phone Number'
            }
            
            self.log_extraction('device_info', 'completed', f"Extracted {len(device_info)} fields")
            
            return {
                'module': 'device_info',
                'status': 'completed',
                'data': device_info,
                'files': 1,
                'size_mb': 0.1
            }
        
        except Exception as e:
            self.log_extraction('device_info', 'failed', str(e))
            return {
                'module': 'device_info',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


class CommunicationsExtractor(BaseExtractor):
    """Extract communications"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract communications"""
        
        is_allowed, reason = self.check_consent('communications')
        
        if not is_allowed:
            return {
                'module': 'communications',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('communications', 'started')
        
        try:
            communications = {
                'sms_count': 150,
                'call_logs_count': 45,
                'whatsapp_messages': 500,
                'telegram_messages': 200,
                'facebook_messages': 300,
                'email_count': 1000
            }
            
            self.log_extraction('communications', 'completed', f"Extracted {sum(communications.values())} items")
            
            return {
                'module': 'communications',
                'status': 'completed',
                'data': communications,
                'files': 150,
                'size_mb': 50
            }
        
        except Exception as e:
            self.log_extraction('communications', 'failed', str(e))
            return {
                'module': 'communications',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


class LocationExtractor(BaseExtractor):
    """Extract location data"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract location"""
        
        is_allowed, reason = self.check_consent('location')
        
        if not is_allowed:
            return {
                'module': 'location',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('location', 'started')
        
        try:
            location = {
                'gps_records': 45,
                'wifi_locations': 120,
                'cell_tower_records': 200,
                'google_timeline': 365,
                'maps_history': 89
            }
            
            self.log_extraction('location', 'completed', f"Extracted {sum(location.values())} records")
            
            return {
                'module': 'location',
                'status': 'completed',
                'data': location,
                'files': 45,
                'size_mb': 10
            }
        
        except Exception as e:
            self.log_extraction('location', 'failed', str(e))
            return {
                'module': 'location',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


class MediaExtractor(BaseExtractor):
    """Extract media"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract media"""
        
        is_allowed, reason = self.check_consent('media')
        
        if not is_allowed:
            return {
                'module': 'media',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('media', 'started')
        
        try:
            media = {
                'photos': 1200,
                'videos': 450,
                'audio_files': 350,
                'documents': 500
            }
            
            self.log_extraction('media', 'completed', f"Extracted {sum(media.values())} files")
            
            return {
                'module': 'media',
                'status': 'completed',
                'data': media,
                'files': 2500,
                'size_mb': 5000
            }
        
        except Exception as e:
            self.log_extraction('media', 'failed', str(e))
            return {
                'module': 'media',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


class SecurityExtractor(BaseExtractor):
    """Extract security data"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract security"""
        
        is_allowed, reason = self.check_consent('security')
        
        if not is_allowed:
            return {
                'module': 'security',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('security', 'started')
        
        try:
            security = {
                'installed_apps': 150,
                'app_permissions': 300,
                'biometric_data': 5,
                'security_logs': 50
            }
            
            self.log_extraction('security', 'completed', f"Extracted {sum(security.values())} items")
            
            return {
                'module': 'security',
                'status': 'completed',
                'data': security,
                'files': 30,
                'size_mb': 5
            }
        
        except Exception as e:
            self.log_extraction('security', 'failed', str(e))
            return {
                'module': 'security',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


class SystemExtractor(BaseExtractor):
    """Extract system data"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract system"""
        
        is_allowed, reason = self.check_consent('system')
        
        if not is_allowed:
            return {
                'module': 'system',
                'status': 'blocked',
                'reason': reason,
                'files': 0,
                'size_mb': 0
            }
        
        self.log_extraction('system', 'started')
        
        try:
            system = {
                'system_logs': 100,
                'crash_logs': 25,
                'event_logs': 200,
                'battery_stats': 50
            }
            
            self.log_extraction('system', 'completed', f"Extracted {sum(system.values())} items")
            
            return {
                'module': 'system',
                'status': 'completed',
                'data': system,
                'files': 100,
                'size_mb': 20
            }
        
        except Exception as e:
            self.log_extraction('system', 'failed', str(e))
            return {
                'module': 'system',
                'status': 'failed',
                'error': str(e),
                'files': 0,
                'size_mb': 0
            }


# ============================================================================
# EXTRACTION ORCHESTRATOR
# ============================================================================

class ExtractionOrchestrator:
    """Orchestrate extraction across all modules"""
    
    def __init__(self, consent_data: Dict):
        """Initialize orchestrator"""
        self.consent_data = consent_data
        self.case_id = consent_data.get('case_id')
        self.consent_level = consent_data.get('consent_level')
        self.modules_allowed = consent_data.get('modules_allowed', [])
        
        # Initialize all extractors
        self.extractors = {
            'device_info': DeviceInfoExtractor(consent_data),
            'communications': CommunicationsExtractor(consent_data),
            'location': LocationExtractor(consent_data),
            'media': MediaExtractor(consent_data),
            'security': SecurityExtractor(consent_data),
            'system': SystemExtractor(consent_data)
        }
    
    def extract_all(self, device_id: str) -> Dict:
        """Extract all modules based on consent"""
        
        results = {
            'case_id': self.case_id,
            'consent_level': self.consent_level,
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'total_files': 0,
            'total_size_mb': 0,
            'extraction_log': []
        }
        
        # Extract each module
        for module_name, extractor in self.extractors.items():
            result = extractor.extract(device_id)
            results['modules'][module_name] = result
            
            # Aggregate results
            if result['status'] == 'completed':
                results['total_files'] += result.get('files', 0)
                results['total_size_mb'] += result.get('size_mb', 0)
            
            # Add to log
            results['extraction_log'].extend(extractor.get_extraction_log())
        
        return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the module"""
    
    # Example consent data (from token)
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com',
        'modules_allowed': ['device_info', 'communications', 'location', 'media'],
        'modules_blocked': ['security', 'system']
    }
    
    # Create orchestrator
    orchestrator = ExtractionOrchestrator(consent_data)
    
    # Extract all modules
    results = orchestrator.extract_all('emulator-5554')
    
    # Display results
    print(f"Case: {results['case_id']}")
    print(f"Consent Level: {results['consent_level']}")
    print(f"Total Files: {results['total_files']}")
    print(f"Total Size: {results['total_size_mb']} MB")
    print(f"\nModule Results:")
    
    for module_name, result in results['modules'].items():
        status = result['status']
        if status == 'completed':
            print(f"  ✅ {module_name}: {result['files']} files ({result['size_mb']} MB)")
        elif status == 'blocked':
            print(f"  ❌ {module_name}: {result['reason']}")
        else:
            print(f"  ⚠️ {module_name}: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    example_usage()
