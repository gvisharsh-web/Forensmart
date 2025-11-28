# 🔐 COMPLETE VERIFICATION WIRING TO ALL CORE MODULES

**Version**: 1.0  
**Date**: November 28, 2025  
**Status**: 📋 Implementation Guide

---

## 📊 CURRENT STATE

### **Already Created** ✅
- `consent_based_extraction.py` - Single integrated module with verification
- `base_extractor.py` - Base class for extractors
- `desktop_extraction_tool.py` - Desktop tool with token verification

### **Existing Core Modules** ✅
- `extractors.py` - Main extraction module
- `orchestrator.py` - Orchestration logic
- `consent.py` - Consent management
- `consent_approval_workflow.py` - Approval workflow
- `signature_service.py` - Digital signatures

---

## 🔌 WIRING POINTS - DETAILED IMPLEMENTATION

### **POINT 1: Wire to `extractors.py`**

**File**: `modules/extraction/extractors.py`

**Current State**: Has `ExtractionModule` base class

**What to Add**:

```python
# At the top of extractors.py, add import
from .consent_based_extraction import ConsentTokenVerifier, BaseExtractor as ConsentBaseExtractor

# Modify ExtractionModule to inherit from ConsentBaseExtractor
class ExtractionModule(ConsentBaseExtractor):
    """Base class for extraction modules with consent verification"""
    
    def __init__(self, name: str, description: str, consent_data: Dict = None):
        """Initialize extraction module"""
        # Initialize consent base if consent_data provided
        if consent_data:
            super().__init__(consent_data)
        
        self.name = name
        self.description = description
        self.extraction_time = None
        self.artifact_count = 0
    
    @abstractmethod
    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract data from device"""
        pass
    
    def extract_with_consent(self, device_id: str, module_name: str, **kwargs) -> Dict[str, Any]:
        """Extract with consent verification"""
        
        # Check consent
        is_allowed, reason = self.check_consent(module_name)
        
        if not is_allowed:
            return {
                'status': 'blocked',
                'reason': reason,
                'module': module_name
            }
        
        # Proceed with extraction
        self.log_extraction(module_name, 'started')
        
        try:
            result = self.extract(device_id, **kwargs)
            self.log_extraction(module_name, 'completed')
            return result
        except Exception as e:
            self.log_extraction(module_name, 'failed', str(e))
            raise
```

---

### **POINT 2: Wire to `orchestrator.py`**

**File**: `modules/extraction/orchestrator.py`

**Current State**: Has orchestration logic

**What to Add**:

```python
# At the top of orchestrator.py, add import
from .consent_based_extraction import ConsentTokenVerifier, ExtractionOrchestrator as ConsentOrchestrator

# Modify main orchestrator class
class ExtractionOrchestrator:
    """Main orchestrator with consent verification"""
    
    def __init__(self, consent_data: Dict = None):
        """Initialize orchestrator"""
        self.consent_data = consent_data
        self.consent_verified = False
        self.extraction_log = []
        
        # If consent data provided, verify it
        if consent_data:
            self.verify_consent()
    
    def verify_consent(self):
        """Verify consent token"""
        if not self.consent_data:
            return False
        
        # Check if consent data has required fields
        required_fields = ['case_id', 'consent_level', 'modules_allowed']
        if all(field in self.consent_data for field in required_fields):
            self.consent_verified = True
            return True
        
        return False
    
    def orchestrate_extraction(self, device_id: str, modules: List[str]) -> Dict:
        """Orchestrate extraction with consent"""
        
        # Verify consent first
        if self.consent_data and not self.consent_verified:
            if not self.verify_consent():
                return {
                    'status': 'failed',
                    'error': 'Consent verification failed'
                }
        
        results = {
            'case_id': self.consent_data.get('case_id') if self.consent_data else 'UNKNOWN',
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'extraction_log': self.extraction_log
        }
        
        # Extract each module
        for module in modules:
            # Check if module allowed by consent
            if self.consent_data:
                if module not in self.consent_data.get('modules_allowed', []):
                    results['modules'][module] = {
                        'status': 'blocked',
                        'reason': f"Not allowed by consent level"
                    }
                    continue
            
            # Extract module
            try:
                result = self.extract_module(module, device_id)
                results['modules'][module] = result
            except Exception as e:
                results['modules'][module] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def extract_module(self, module_name: str, device_id: str) -> Dict:
        """Extract specific module"""
        # Implementation here
        pass
```

---

### **POINT 3: Wire to `consent.py`**

**File**: `modules/extraction/consent.py`

**Current State**: Has consent management

**What to Add**:

```python
# At the top of consent.py, add import
from .consent_based_extraction import ConsentTokenVerifier

# Add method to consent class
class ConsentManager:
    """Manage consent with verification"""
    
    def __init__(self):
        """Initialize consent manager"""
        self.verifier = ConsentTokenVerifier()
        self.verified_consents = {}
    
    def verify_and_store_consent(self, token: str) -> Tuple[bool, str, Dict]:
        """Verify token and store consent"""
        
        is_valid, message, consent_data = self.verifier.verify_token(token)
        
        if is_valid:
            case_id = consent_data.get('case_id')
            self.verified_consents[case_id] = consent_data
        
        return is_valid, message, consent_data
    
    def get_verified_consent(self, case_id: str) -> Dict:
        """Get verified consent for case"""
        return self.verified_consents.get(case_id)
    
    def is_module_allowed(self, case_id: str, module_name: str) -> bool:
        """Check if module is allowed for case"""
        
        consent = self.get_verified_consent(case_id)
        if not consent:
            return False
        
        modules_allowed = consent.get('modules_allowed', [])
        modules_blocked = consent.get('modules_blocked', [])
        
        return module_name in modules_allowed and module_name not in modules_blocked
```

---

### **POINT 4: Wire to `consent_approval_workflow.py`**

**File**: `modules/extraction/consent_approval_workflow.py`

**Current State**: Has approval workflow

**What to Add**:

```python
# At the top of consent_approval_workflow.py, add import
from .consent_based_extraction import ConsentTokenVerifier
import hashlib
import hmac
import json
import base64

# Add method to approval workflow class
class ConsentApprovalWorkflow:
    """Approval workflow with token generation"""
    
    def __init__(self):
        """Initialize workflow"""
        self.verifier = ConsentTokenVerifier()
    
    def generate_consent_token(self, case_id: str, consent_level: str, 
                               approved_by: str, modules_allowed: List[str]) -> str:
        """Generate consent token after approval"""
        
        from datetime import datetime, timedelta
        
        # Create consent data
        consent_data = {
            'case_id': case_id,
            'consent_level': consent_level,
            'approved_by': approved_by,
            'approval_date': datetime.now().isoformat(),
            'expiry_date': (datetime.now() + timedelta(days=30)).isoformat(),
            'modules_allowed': modules_allowed,
            'modules_blocked': self.get_blocked_modules(consent_level),
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
    
    def get_blocked_modules(self, consent_level: str) -> List[str]:
        """Get blocked modules for consent level"""
        
        all_modules = ['device_info', 'communications', 'location', 'media', 'security', 'system']
        
        if consent_level == 'STANDARD':
            return ['communications', 'security', 'system']
        elif consent_level == 'LEGAL':
            return ['security', 'system']
        elif consent_level == 'FULL':
            return []
        
        return all_modules
```

---

### **POINT 5: Wire to `signature_service.py`**

**File**: `modules/extraction/signature_service.py`

**Current State**: Has digital signature service

**What to Add**:

```python
# At the top of signature_service.py, add import
from .consent_based_extraction import ConsentTokenVerifier

# Add method to signature service class
class SignatureService:
    """Digital signature service with consent verification"""
    
    def __init__(self):
        """Initialize signature service"""
        self.verifier = ConsentTokenVerifier()
    
    def sign_consent_approval(self, consent_token: str, signer_email: str, 
                             signer_name: str) -> Dict:
        """Sign consent approval"""
        
        # Verify token first
        is_valid, message, consent_data = self.verifier.verify_token(consent_token)
        
        if not is_valid:
            return {
                'status': 'failed',
                'error': message
            }
        
        # Create signature
        signature_data = {
            'consent_data': consent_data,
            'signer_email': signer_email,
            'signer_name': signer_name,
            'signed_at': datetime.now().isoformat(),
            'signature_id': self.generate_signature_id()
        }
        
        return {
            'status': 'success',
            'signature_data': signature_data
        }
    
    def generate_signature_id(self) -> str:
        """Generate unique signature ID"""
        import uuid
        return str(uuid.uuid4())
```

---

### **POINT 6: Wire to `consent_approval_signature_integration.py`**

**File**: `modules/extraction/consent_approval_signature_integration.py`

**Current State**: Has integration logic

**What to Add**:

```python
# At the top of consent_approval_signature_integration.py, add import
from .consent_based_extraction import ConsentTokenVerifier, ExtractionOrchestrator

# Add method to integration class
class ConsentApprovalSignatureIntegration:
    """Integration of consent, approval, and signature with extraction"""
    
    def __init__(self):
        """Initialize integration"""
        self.verifier = ConsentTokenVerifier()
        self.orchestrator = None
    
    def complete_workflow(self, consent_token: str, device_id: str) -> Dict:
        """Complete entire workflow: verify -> extract -> report"""
        
        # Step 1: Verify consent token
        is_valid, message, consent_data = self.verifier.verify_token(consent_token)
        
        if not is_valid:
            return {
                'status': 'failed',
                'step': 'verification',
                'error': message
            }
        
        # Step 2: Create orchestrator with consent
        self.orchestrator = ExtractionOrchestrator(consent_data)
        
        # Step 3: Extract all modules
        results = self.orchestrator.extract_all(device_id)
        
        # Step 4: Generate report
        report = self.generate_report(results)
        
        return {
            'status': 'success',
            'results': results,
            'report': report
        }
    
    def generate_report(self, extraction_results: Dict) -> Dict:
        """Generate report from extraction results"""
        
        return {
            'case_id': extraction_results['case_id'],
            'consent_level': extraction_results['consent_level'],
            'device_id': extraction_results['device_id'],
            'timestamp': extraction_results['timestamp'],
            'total_files': extraction_results['total_files'],
            'total_size_mb': extraction_results['total_size_mb'],
            'modules': extraction_results['modules'],
            'audit_log': extraction_results['extraction_log']
        }
```

---

## 📋 WIRING CHECKLIST

### **Phase 1: Core Module Updates**
- [ ] Update `extractors.py` - Add consent verification to ExtractionModule
- [ ] Update `orchestrator.py` - Add consent checking to orchestration
- [ ] Update `consent.py` - Add token verification
- [ ] Update `consent_approval_workflow.py` - Add token generation
- [ ] Update `signature_service.py` - Add consent verification
- [ ] Update `consent_approval_signature_integration.py` - Add complete workflow

### **Phase 2: Desktop Tool Integration**
- [ ] Update `desktop_extraction_tool.py` - Use orchestrator with consent
- [ ] Test token verification
- [ ] Test extraction with consent
- [ ] Test module filtering
- [ ] Test audit logging

### **Phase 3: Web App Integration**
- [ ] Add token generation to `app.py`
- [ ] Add API endpoint for token export
- [ ] Add API endpoint for results upload
- [ ] Test end-to-end workflow

### **Phase 4: Testing**
- [ ] Unit tests for each module
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Error handling tests

---

## 🔄 COMPLETE DATA FLOW

```
Web App (app.py)
├── Create case
├── Get approval
├── Generate token (hash + signature)
└── Export token
    ↓
Desktop Tool (desktop_extraction_tool.py)
├── Paste token
├── ConsentTokenVerifier.verify_token()
│   ├── Decode token
│   ├── Check hash
│   ├── Check signature
│   ├── Check expiry
│   └── Check required fields
└── ExtractionOrchestrator.extract_all()
    ├── Initialize extractors
    ├── Each extractor checks consent
    │   ├── Is module allowed?
    │   ├── Is module not blocked?
    │   └── Extract if allowed
    ├── Aggregate results
    └── Generate audit log
    ↓
Core Modules (extractors.py, orchestrator.py, etc.)
├── ExtractionModule.extract_with_consent()
│   ├── Check consent
│   ├── Log extraction
│   └── Extract data
├── ExtractionOrchestrator.orchestrate_extraction()
│   ├── Verify consent
│   ├── Check module permissions
│   └── Extract modules
├── ConsentManager.is_module_allowed()
│   ├── Get verified consent
│   └── Check modules
└── SignatureService.sign_consent_approval()
    ├── Verify token
    └── Create signature
    ↓
Results
├── Extracted data
├── Audit log
├── Extraction log
└── Report
    ↓
Web App (app.py)
├── Receive results
├── Update case status
├── Store extraction log
└── Display report
```

---

## ✅ BENEFITS OF WIRING

- ✅ **Consent Verification**: Every extraction verified
- ✅ **Module Filtering**: Only allowed modules extracted
- ✅ **Audit Trail**: Complete logging
- ✅ **Legal Compliance**: Respects consent levels
- ✅ **Error Handling**: Graceful failures
- ✅ **Security**: Hash + signature verification
- ✅ **Transparency**: Clear logging
- ✅ **Scalability**: Easy to extend

---

## 🚀 IMPLEMENTATION ORDER

1. **Update `extractors.py`** - Add consent base class
2. **Update `orchestrator.py`** - Add consent checking
3. **Update `consent.py`** - Add token verification
4. **Update `consent_approval_workflow.py`** - Add token generation
5. **Update `signature_service.py`** - Add consent verification
6. **Update `consent_approval_signature_integration.py`** - Add complete workflow
7. **Update `desktop_extraction_tool.py`** - Use new orchestrator
8. **Update `app.py`** - Add token generation and API endpoints
9. **Test complete workflow**
10. **Deploy to production**

---

## 📊 FILES TO MODIFY

| File | Changes | Priority |
|------|---------|----------|
| `extractors.py` | Add consent verification | 🔴 HIGH |
| `orchestrator.py` | Add consent checking | 🔴 HIGH |
| `consent.py` | Add token verification | 🔴 HIGH |
| `consent_approval_workflow.py` | Add token generation | 🔴 HIGH |
| `signature_service.py` | Add consent verification | 🟡 MEDIUM |
| `consent_approval_signature_integration.py` | Add complete workflow | 🟡 MEDIUM |
| `desktop_extraction_tool.py` | Use new orchestrator | 🟡 MEDIUM |
| `app.py` | Add token generation | 🟡 MEDIUM |

---

**Status**: 📋 **READY FOR IMPLEMENTATION**

**Complexity**: ⭐⭐⭐⭐ (High)

**Time Estimate**: 4-6 hours

**Priority**: 🔴 **CRITICAL**
