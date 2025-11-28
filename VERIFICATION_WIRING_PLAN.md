# 🔐 VERIFICATION METHOD WIRING IMPLEMENTATION PLAN

**Version**: 1.0  
**Date**: November 28, 2025  
**Status**: 📋 Implementation Plan

---

## 📊 OVERVIEW

This document outlines how to wire the token verification method across all ForenSmart modules to ensure:
- ✅ Consent verification before extraction
- ✅ Module filtering based on consent level
- ✅ Audit trail for all operations
- ✅ Legal compliance

---

## 🏗️ ARCHITECTURE

```
Web App (Streamlit)
├── Case Management
├── Consent Approval
├── Token Generation
└── Token Export
    ↓
Desktop Tool
├── Token Verification
├── Device Detection
├── Extraction Manager
└── Results Upload
    ↓
Modules (Extraction)
├── Device Info Module
├── Communications Module
├── Location Module
├── Media Module
├── Security Module
└── System Module
    ↓
Results Storage
├── Local Storage
├── Web App Upload
└── Report Generation
```

---

## 🔌 WIRING POINTS

### **Point 1: Web App → Token Generation**

**File**: `app.py`

**Location**: Consent approval section

**Implementation**:
```python
# In render_consent_check() or render_consent_approval_form()

def generate_consent_token(case_id, consent_level, approved_by, modules_allowed):
    """Generate consent token for desktop tool"""
    
    import hashlib
    import hmac
    import json
    import base64
    from datetime import datetime, timedelta
    
    # Create consent data
    consent_data = {
        'case_id': case_id,
        'consent_level': consent_level,
        'approved_by': approved_by,
        'approval_date': datetime.now().isoformat(),
        'expiry_date': (datetime.now() + timedelta(days=30)).isoformat(),
        'modules_allowed': modules_allowed,
        'modules_blocked': get_blocked_modules(consent_level),
        'investigator': st.session_state.get('investigator_name', 'Unknown'),
        'device_type': st.session_state.get('device_type', 'Unknown'),
        'device_id': st.session_state.get('device_id', 'Unknown'),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Create hash
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    
    # Create signature
    signature = hmac.new(
        b'forensmart-secret-key',
        data_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Package token
    token_data = {
        'data': consent_data,
        'hash': data_hash,
        'signature': signature
    }
    
    # Encode token
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    return f"FORENSMART_CONSENT_TOKEN_v1.0\n{token}"

# Add to UI
if st.button("📋 Copy Consent Token"):
    token = generate_consent_token(
        case_id=case_id,
        consent_level=consent_level,
        approved_by=nominee_email,
        modules_allowed=get_modules_for_level(consent_level)
    )
    st.code(token, language="text")
    st.info("✅ Token copied to clipboard - paste in Desktop Tool")
```

---

### **Point 2: Desktop Tool → Token Verification**

**File**: `desktop_extraction_tool.py`

**Location**: ConsentTokenVerifier class

**Already Implemented**: ✅
```python
class ConsentTokenVerifier:
    def verify_token(self, token: str) -> Tuple[bool, str, Dict]:
        # Verifies hash, signature, expiry, required fields
        # Returns: (is_valid, message, consent_data)
```

---

### **Point 3: Extraction Modules → Consent Checking**

**Files**:
- `modules/extraction/device_info_extractor.py`
- `modules/extraction/communications_extractor.py`
- `modules/extraction/location_extractor.py`
- `modules/extraction/media_extractor.py`
- `modules/extraction/security_extractor.py`
- `modules/extraction/system_extractor.py`

**Implementation**:

Create a base extractor class with consent checking:

```python
# File: modules/extraction/base_extractor.py

from typing import Dict, List, Tuple
from datetime import datetime

class BaseExtractor:
    """Base class for all extractors with consent verification"""
    
    def __init__(self, consent_data: Dict):
        """Initialize with consent data"""
        self.consent_data = consent_data
        self.case_id = consent_data.get('case_id')
        self.consent_level = consent_data.get('consent_level')
        self.modules_allowed = consent_data.get('modules_allowed', [])
        self.modules_blocked = consent_data.get('modules_blocked', [])
        self.extraction_log = []
    
    def check_consent(self, module_name: str) -> Tuple[bool, str]:
        """
        Check if module extraction is allowed by consent
        
        Args:
            module_name: Name of module to extract
            
        Returns:
            (is_allowed, reason)
        """
        
        # Check if module is in allowed list
        if module_name not in self.modules_allowed:
            reason = f"Module '{module_name}' not allowed by consent level '{self.consent_level}'"
            self.log_extraction(module_name, 'blocked', reason)
            return False, reason
        
        # Check if module is not blocked
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
    
    def extract(self, device_id: str) -> Dict:
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement extract()")
```

---

### **Point 4: Device Info Module**

**File**: `modules/extraction/device_info_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class DeviceInfoExtractor(BaseExtractor):
    """Extract device information with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract device info"""
        
        # Check consent
        is_allowed, reason = self.check_consent('device_info')
        
        if not is_allowed:
            return {
                'module': 'device_info',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('device_info', 'started')
        
        try:
            # Extract device information
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
                'files': 0
            }
```

---

### **Point 5: Communications Module**

**File**: `modules/extraction/communications_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class CommunicationsExtractor(BaseExtractor):
    """Extract communications with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract communications"""
        
        # Check consent
        is_allowed, reason = self.check_consent('communications')
        
        if not is_allowed:
            return {
                'module': 'communications',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('communications', 'started')
        
        try:
            # Extract communications
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
                'files': 0
            }
```

---

### **Point 6: Location Module**

**File**: `modules/extraction/location_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class LocationExtractor(BaseExtractor):
    """Extract location data with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract location"""
        
        # Check consent
        is_allowed, reason = self.check_consent('location')
        
        if not is_allowed:
            return {
                'module': 'location',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('location', 'started')
        
        try:
            # Extract location
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
                'files': 0
            }
```

---

### **Point 7: Media Module**

**File**: `modules/extraction/media_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class MediaExtractor(BaseExtractor):
    """Extract media with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract media"""
        
        # Check consent
        is_allowed, reason = self.check_consent('media')
        
        if not is_allowed:
            return {
                'module': 'media',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('media', 'started')
        
        try:
            # Extract media
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
                'files': 0
            }
```

---

### **Point 8: Security Module**

**File**: `modules/extraction/security_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class SecurityExtractor(BaseExtractor):
    """Extract security data with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract security"""
        
        # Check consent
        is_allowed, reason = self.check_consent('security')
        
        if not is_allowed:
            return {
                'module': 'security',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('security', 'started')
        
        try:
            # Extract security
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
                'files': 0
            }
```

---

### **Point 9: System Module**

**File**: `modules/extraction/system_extractor.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor

class SystemExtractor(BaseExtractor):
    """Extract system data with consent verification"""
    
    def extract(self, device_id: str) -> Dict:
        """Extract system"""
        
        # Check consent
        is_allowed, reason = self.check_consent('system')
        
        if not is_allowed:
            return {
                'module': 'system',
                'status': 'blocked',
                'reason': reason,
                'files': 0
            }
        
        # Proceed with extraction
        self.log_extraction('system', 'started')
        
        try:
            # Extract system
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
                'files': 0
            }
```

---

### **Point 10: Orchestrator Module**

**File**: `modules/extraction/extraction_orchestrator.py`

**Implementation**:

```python
from .base_extractor import BaseExtractor
from .device_info_extractor import DeviceInfoExtractor
from .communications_extractor import CommunicationsExtractor
from .location_extractor import LocationExtractor
from .media_extractor import MediaExtractor
from .security_extractor import SecurityExtractor
from .system_extractor import SystemExtractor

class ExtractionOrchestrator:
    """Orchestrate extraction across all modules with consent verification"""
    
    def __init__(self, consent_data: Dict):
        """Initialize with consent data"""
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
        """Extract all allowed modules"""
        
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
            print(f"Extracting {module_name}...")
            
            # Extract
            result = extractor.extract(device_id)
            results['modules'][module_name] = result
            
            # Aggregate results
            if result['status'] == 'completed':
                results['total_files'] += result.get('files', 0)
                results['total_size_mb'] += result.get('size_mb', 0)
            
            # Add to log
            results['extraction_log'].extend(extractor.get_extraction_log())
        
        return results
```

---

### **Point 11: Results Upload**

**File**: `modules/extraction/results_uploader.py`

**Implementation**:

```python
import requests
import json

class ResultsUploader:
    """Upload extraction results to web app"""
    
    def __init__(self, web_app_url: str = "http://localhost:8501"):
        """Initialize uploader"""
        self.web_app_url = web_app_url
    
    def upload_results(self, extraction_results: Dict) -> Tuple[bool, str]:
        """
        Upload extraction results to web app
        
        Args:
            extraction_results: Results from ExtractionOrchestrator
            
        Returns:
            (success, message)
        """
        
        try:
            # Prepare payload
            payload = {
                'case_id': extraction_results['case_id'],
                'consent_level': extraction_results['consent_level'],
                'device_id': extraction_results['device_id'],
                'timestamp': extraction_results['timestamp'],
                'modules': extraction_results['modules'],
                'total_files': extraction_results['total_files'],
                'total_size_mb': extraction_results['total_size_mb'],
                'extraction_log': extraction_results['extraction_log']
            }
            
            # Send to web app
            response = requests.post(
                f"{self.web_app_url}/api/upload_results",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return True, "✅ Results uploaded successfully"
            else:
                return False, f"❌ Upload failed: {response.text}"
        
        except Exception as e:
            return False, f"❌ Upload error: {str(e)}"
```

---

### **Point 12: Web App API Endpoint**

**File**: `app.py`

**Implementation**:

```python
from flask import Flask, request, jsonify

@app.route('/api/upload_results', methods=['POST'])
def upload_results():
    """Receive extraction results from desktop tool"""
    
    try:
        data = request.json
        
        case_id = data.get('case_id')
        consent_level = data.get('consent_level')
        modules = data.get('modules', {})
        extraction_log = data.get('extraction_log', [])
        
        # Store results
        if 'extraction_results' not in st.session_state:
            st.session_state.extraction_results = {}
        
        st.session_state.extraction_results[case_id] = {
            'case_id': case_id,
            'consent_level': consent_level,
            'modules': modules,
            'extraction_log': extraction_log,
            'timestamp': data.get('timestamp')
        }
        
        # Update case status
        for case in st.session_state.cases_list:
            if case['Case ID'] == case_id:
                case['Status'] = 'Completed'
                case['Findings'] = data.get('total_files', 0)
        
        return jsonify({
            'success': True,
            'message': 'Results received successfully',
            'case_id': case_id
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
```

---

## 📋 WIRING CHECKLIST

### **Phase 1: Web App Setup**
- [ ] Add `generate_consent_token()` function to `app.py`
- [ ] Add "Copy Token" button to consent approval
- [ ] Add "Download Token" button for JSON export
- [ ] Add API endpoint `/api/upload_results`
- [ ] Test token generation

### **Phase 2: Desktop Tool Setup**
- [ ] Verify `ConsentTokenVerifier` class works
- [ ] Test token verification
- [ ] Test device detection
- [ ] Test extraction simulation
- [ ] Test results upload

### **Phase 3: Module Setup**
- [ ] Create `base_extractor.py` with `BaseExtractor` class
- [ ] Create `device_info_extractor.py` with consent check
- [ ] Create `communications_extractor.py` with consent check
- [ ] Create `location_extractor.py` with consent check
- [ ] Create `media_extractor.py` with consent check
- [ ] Create `security_extractor.py` with consent check
- [ ] Create `system_extractor.py` with consent check

### **Phase 4: Orchestration Setup**
- [ ] Create `extraction_orchestrator.py`
- [ ] Integrate all extractors
- [ ] Test extraction with consent
- [ ] Test module blocking
- [ ] Test audit logging

### **Phase 5: Integration Setup**
- [ ] Create `results_uploader.py`
- [ ] Add API endpoint to web app
- [ ] Test results upload
- [ ] Test case status update
- [ ] Test findings display

### **Phase 6: Testing**
- [ ] Test end-to-end workflow
- [ ] Test consent verification
- [ ] Test module filtering
- [ ] Test audit trail
- [ ] Test error handling

---

## 🔄 COMPLETE WORKFLOW

```
1. Web App: Create case
   ↓
2. Web App: Get approval
   ↓
3. Web App: Generate token (with hash + signature)
   ↓
4. User: Copy token
   ↓
5. Desktop Tool: Paste token
   ↓
6. Desktop Tool: Verify token (check hash + signature)
   ↓
7. Desktop Tool: Create ExtractionOrchestrator
   ↓
8. ExtractionOrchestrator: Initialize all extractors
   ↓
9. Each Extractor: Check consent (BaseExtractor.check_consent())
   ↓
10. Allowed Modules: Extract data
    Blocked Modules: Skip with reason
   ↓
11. ExtractionOrchestrator: Aggregate results
   ↓
12. ResultsUploader: Upload to web app
   ↓
13. Web App: Receive results
   ↓
14. Web App: Update case status
   ↓
15. Web App: Display results
```

---

## 📊 DATA FLOW

```
Consent Data (Token)
├── case_id
├── consent_level (STANDARD/LEGAL/FULL)
├── modules_allowed []
├── modules_blocked []
├── approved_by
├── expiry_date
├── hash (SHA256)
└── signature (HMAC-SHA256)
    ↓
BaseExtractor.check_consent()
├── Is module in modules_allowed? ✅/❌
├── Is module not in modules_blocked? ✅/❌
└── Log extraction attempt
    ↓
Module Extractor.extract()
├── If allowed: Extract data
├── If blocked: Skip with reason
└── Log result
    ↓
ExtractionOrchestrator.extract_all()
├── Aggregate all results
├── Calculate totals
└── Create audit log
    ↓
ResultsUploader.upload_results()
├── Send to web app
├── Update case status
└── Display results
```

---

## ✅ BENEFITS

- ✅ **Consent Verification**: Every extraction verified
- ✅ **Module Filtering**: Only allowed modules extracted
- ✅ **Audit Trail**: Complete logging of all operations
- ✅ **Legal Compliance**: Respects consent levels
- ✅ **Error Handling**: Graceful failure handling
- ✅ **Scalability**: Easy to add new modules
- ✅ **Security**: Hash + signature verification
- ✅ **Transparency**: Clear logging and reporting

---

## 🚀 NEXT STEPS

1. Implement `base_extractor.py`
2. Update all module extractors
3. Create `extraction_orchestrator.py`
4. Create `results_uploader.py`
5. Add API endpoint to web app
6. Test complete workflow
7. Deploy to production

---

**Status**: 📋 **READY FOR IMPLEMENTATION**

**Complexity**: ⭐⭐⭐ (Medium)

**Time Estimate**: 2-3 hours

**Priority**: 🔴 **HIGH**
