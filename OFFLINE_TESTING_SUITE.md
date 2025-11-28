# 🧪 OFFLINE TESTING SUITE

**Version**: 1.0  
**Date**: November 28, 2025  
**Status**: 📋 Testing Plan

---

## 📊 OVERVIEW

Complete offline testing suite for ForenSmart verification system:
- ✅ No internet required
- ✅ No external dependencies
- ✅ Hash-based verification (already implemented)
- ✅ Local device simulation
- ✅ Complete workflow testing
- ✅ Error handling testing

---

## 🎯 TESTING STRATEGY

### **What We Test**
1. ✅ Token generation (hash + signature)
2. ✅ Token verification (hash + signature check)
3. ✅ Consent checking
4. ✅ Module filtering
5. ✅ Extraction with consent
6. ✅ Audit logging
7. ✅ Error handling
8. ✅ End-to-end workflow

### **What We Don't Need**
- ❌ Internet connection
- ❌ Real devices
- ❌ External APIs
- ❌ Cloud services
- ❌ Database (use in-memory)

---

## 🧪 TEST SUITE 1: TOKEN VERIFICATION

### **Test 1.1: Valid Token**

```python
# Test: Generate and verify valid token

from modules.extraction.consent_based_extraction import ConsentTokenVerifier
import hashlib
import hmac
import json
import base64

def test_valid_token():
    """Test valid token verification"""
    
    # Create consent data
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com',
        'modules_allowed': ['device_info', 'communications', 'location', 'media'],
        'modules_blocked': ['security', 'system']
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
    
    # Verify token
    verifier = ConsentTokenVerifier()
    is_valid, message, result = verifier.verify_token(token)
    
    # Assert
    assert is_valid == True, f"Token verification failed: {message}"
    assert result['case_id'] == 'CASE-001'
    assert result['consent_level'] == 'LEGAL'
    
    print("✅ Test 1.1 PASSED: Valid token verified")
    return True
```

### **Test 1.2: Tampered Token (Hash Mismatch)**

```python
def test_tampered_token_hash():
    """Test tampered token detection (hash mismatch)"""
    
    # Create valid token first
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com',
        'modules_allowed': ['device_info', 'communications']
    }
    
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    signature = hmac.new(
        b'forensmart-secret-key',
        data_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Tamper with hash
    tampered_hash = 'abc123def456'  # Wrong hash
    
    token_data = {
        'data': consent_data,
        'hash': tampered_hash,  # Tampered
        'signature': signature
    }
    
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    # Verify token
    verifier = ConsentTokenVerifier()
    is_valid, message, result = verifier.verify_token(token)
    
    # Assert
    assert is_valid == False, "Tampered token should be rejected"
    assert "Hash mismatch" in message
    
    print("✅ Test 1.2 PASSED: Tampered token detected (hash)")
    return True
```

### **Test 1.3: Invalid Signature**

```python
def test_invalid_signature():
    """Test invalid signature detection"""
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com'
    }
    
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    
    # Wrong signature
    wrong_signature = 'xyz789abc123'
    
    token_data = {
        'data': consent_data,
        'hash': data_hash,
        'signature': wrong_signature  # Wrong
    }
    
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    # Verify token
    verifier = ConsentTokenVerifier()
    is_valid, message, result = verifier.verify_token(token)
    
    # Assert
    assert is_valid == False, "Invalid signature should be rejected"
    assert "Signature mismatch" in message
    
    print("✅ Test 1.3 PASSED: Invalid signature detected")
    return True
```

### **Test 1.4: Expired Token**

```python
def test_expired_token():
    """Test expired token detection"""
    
    from datetime import datetime, timedelta
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com',
        'expiry_date': (datetime.now() - timedelta(days=1)).isoformat()  # Expired
    }
    
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    signature = hmac.new(
        b'forensmart-secret-key',
        data_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    token_data = {
        'data': consent_data,
        'hash': data_hash,
        'signature': signature
    }
    
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    # Verify token
    verifier = ConsentTokenVerifier()
    is_valid, message, result = verifier.verify_token(token)
    
    # Assert
    assert is_valid == False, "Expired token should be rejected"
    assert "expired" in message.lower()
    
    print("✅ Test 1.4 PASSED: Expired token detected")
    return True
```

---

## 🧪 TEST SUITE 2: CONSENT CHECKING

### **Test 2.1: Module Allowed**

```python
def test_module_allowed():
    """Test module allowed by consent"""
    
    from modules.extraction.consent_based_extraction import DeviceInfoExtractor
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'modules_allowed': ['device_info', 'communications'],
        'modules_blocked': ['security', 'system']
    }
    
    extractor = DeviceInfoExtractor(consent_data)
    is_allowed, reason = extractor.check_consent('device_info')
    
    # Assert
    assert is_allowed == True, "Module should be allowed"
    
    print("✅ Test 2.1 PASSED: Module allowed by consent")
    return True
```

### **Test 2.2: Module Blocked**

```python
def test_module_blocked():
    """Test module blocked by consent"""
    
    from modules.extraction.consent_based_extraction import SecurityExtractor
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'STANDARD',
        'modules_allowed': ['device_info', 'location', 'media'],
        'modules_blocked': ['communications', 'security', 'system']
    }
    
    extractor = SecurityExtractor(consent_data)
    is_allowed, reason = extractor.check_consent('security')
    
    # Assert
    assert is_allowed == False, "Module should be blocked"
    assert 'blocked' in reason.lower()
    
    print("✅ Test 2.2 PASSED: Module blocked by consent")
    return True
```

---

## 🧪 TEST SUITE 3: EXTRACTION WITH CONSENT

### **Test 3.1: Extract Allowed Module**

```python
def test_extract_allowed_module():
    """Test extraction of allowed module"""
    
    from modules.extraction.consent_based_extraction import DeviceInfoExtractor
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'modules_allowed': ['device_info', 'communications', 'location', 'media'],
        'modules_blocked': ['security', 'system']
    }
    
    extractor = DeviceInfoExtractor(consent_data)
    result = extractor.extract('device-001')
    
    # Assert
    assert result['status'] == 'completed', "Extraction should complete"
    assert result['module'] == 'device_info'
    assert result['files'] > 0
    
    print("✅ Test 3.1 PASSED: Allowed module extracted")
    return True
```

### **Test 3.2: Extract Blocked Module**

```python
def test_extract_blocked_module():
    """Test extraction of blocked module"""
    
    from modules.extraction.consent_based_extraction import SecurityExtractor
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'STANDARD',
        'modules_allowed': ['device_info', 'location', 'media'],
        'modules_blocked': ['communications', 'security', 'system']
    }
    
    extractor = SecurityExtractor(consent_data)
    result = extractor.extract('device-001')
    
    # Assert
    assert result['status'] == 'blocked', "Extraction should be blocked"
    assert 'reason' in result
    
    print("✅ Test 3.2 PASSED: Blocked module not extracted")
    return True
```

---

## 🧪 TEST SUITE 4: ORCHESTRATION

### **Test 4.1: Extract All Allowed Modules**

```python
def test_orchestrate_all_allowed():
    """Test orchestration with all modules allowed"""
    
    from modules.extraction.consent_based_extraction import ExtractionOrchestrator
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'FULL',
        'modules_allowed': ['device_info', 'communications', 'location', 'media', 'security', 'system'],
        'modules_blocked': []
    }
    
    orchestrator = ExtractionOrchestrator(consent_data)
    results = orchestrator.extract_all('device-001')
    
    # Assert
    assert results['case_id'] == 'CASE-001'
    assert len(results['modules']) == 6
    assert all(m['status'] == 'completed' for m in results['modules'].values())
    assert results['total_files'] > 0
    
    print("✅ Test 4.1 PASSED: All modules extracted")
    return True
```

### **Test 4.2: Extract Partial Modules (LEGAL)**

```python
def test_orchestrate_legal_level():
    """Test orchestration with LEGAL consent level"""
    
    from modules.extraction.consent_based_extraction import ExtractionOrchestrator
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'modules_allowed': ['device_info', 'communications', 'location', 'media'],
        'modules_blocked': ['security', 'system']
    }
    
    orchestrator = ExtractionOrchestrator(consent_data)
    results = orchestrator.extract_all('device-001')
    
    # Assert
    assert results['modules']['device_info']['status'] == 'completed'
    assert results['modules']['communications']['status'] == 'completed'
    assert results['modules']['security']['status'] == 'blocked'
    assert results['modules']['system']['status'] == 'blocked'
    
    print("✅ Test 4.2 PASSED: LEGAL level extraction correct")
    return True
```

### **Test 4.3: Extract Minimal Modules (STANDARD)**

```python
def test_orchestrate_standard_level():
    """Test orchestration with STANDARD consent level"""
    
    from modules.extraction.consent_based_extraction import ExtractionOrchestrator
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'STANDARD',
        'modules_allowed': ['device_info', 'location', 'media'],
        'modules_blocked': ['communications', 'security', 'system']
    }
    
    orchestrator = ExtractionOrchestrator(consent_data)
    results = orchestrator.extract_all('device-001')
    
    # Assert
    assert results['modules']['device_info']['status'] == 'completed'
    assert results['modules']['communications']['status'] == 'blocked'
    assert results['modules']['security']['status'] == 'blocked'
    
    print("✅ Test 4.3 PASSED: STANDARD level extraction correct")
    return True
```

---

## 🧪 TEST SUITE 5: AUDIT LOGGING

### **Test 5.1: Extraction Logging**

```python
def test_extraction_logging():
    """Test extraction logging"""
    
    from modules.extraction.consent_based_extraction import DeviceInfoExtractor
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'modules_allowed': ['device_info'],
        'modules_blocked': []
    }
    
    extractor = DeviceInfoExtractor(consent_data)
    extractor.extract('device-001')
    
    log = extractor.get_extraction_log()
    
    # Assert
    assert len(log) > 0, "Log should have entries"
    assert log[0]['case_id'] == 'CASE-001'
    assert log[0]['module'] == 'device_info'
    
    print("✅ Test 5.1 PASSED: Extraction logging works")
    return True
```

### **Test 5.2: Complete Audit Trail**

```python
def test_complete_audit_trail():
    """Test complete audit trail"""
    
    from modules.extraction.consent_based_extraction import ExtractionOrchestrator
    
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'modules_allowed': ['device_info', 'communications'],
        'modules_blocked': ['security']
    }
    
    orchestrator = ExtractionOrchestrator(consent_data)
    results = orchestrator.extract_all('device-001')
    
    # Assert
    assert 'extraction_log' in results
    assert len(results['extraction_log']) > 0
    
    # Check log entries
    for entry in results['extraction_log']:
        assert 'timestamp' in entry
        assert 'case_id' in entry
        assert 'module' in entry
        assert 'status' in entry
    
    print("✅ Test 5.2 PASSED: Complete audit trail recorded")
    return True
```

---

## 🧪 TEST SUITE 6: ERROR HANDLING

### **Test 6.1: Missing Required Fields**

```python
def test_missing_required_fields():
    """Test missing required fields in token"""
    
    from modules.extraction.consent_based_extraction import ConsentTokenVerifier
    
    # Missing 'modules_allowed'
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com'
        # Missing: modules_allowed
    }
    
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    signature = hmac.new(
        b'forensmart-secret-key',
        data_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    token_data = {
        'data': consent_data,
        'hash': data_hash,
        'signature': signature
    }
    
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    verifier = ConsentTokenVerifier()
    is_valid, message, result = verifier.verify_token(token)
    
    # Assert
    assert is_valid == False, "Should reject token with missing fields"
    assert "Missing fields" in message
    
    print("✅ Test 6.1 PASSED: Missing fields detected")
    return True
```

---

## 🧪 TEST SUITE 7: END-TO-END WORKFLOW

### **Test 7.1: Complete Workflow**

```python
def test_complete_workflow():
    """Test complete end-to-end workflow"""
    
    from modules.extraction.consent_based_extraction import (
        ConsentTokenVerifier,
        ExtractionOrchestrator
    )
    
    # Step 1: Create consent data
    consent_data = {
        'case_id': 'CASE-001',
        'consent_level': 'LEGAL',
        'approved_by': 'nominee@example.com',
        'modules_allowed': ['device_info', 'communications', 'location', 'media'],
        'modules_blocked': ['security', 'system']
    }
    
    # Step 2: Create token
    data_json = json.dumps(consent_data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()
    signature = hmac.new(
        b'forensmart-secret-key',
        data_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    token_data = {
        'data': consent_data,
        'hash': data_hash,
        'signature': signature
    }
    
    token = base64.b64encode(
        json.dumps(token_data).encode()
    ).decode()
    
    # Step 3: Verify token
    verifier = ConsentTokenVerifier()
    is_valid, message, verified_consent = verifier.verify_token(token)
    
    assert is_valid == True, "Token verification failed"
    
    # Step 4: Extract with consent
    orchestrator = ExtractionOrchestrator(verified_consent)
    results = orchestrator.extract_all('device-001')
    
    # Step 5: Verify results
    assert results['case_id'] == 'CASE-001'
    assert results['total_files'] > 0
    assert results['modules']['device_info']['status'] == 'completed'
    assert results['modules']['security']['status'] == 'blocked'
    
    print("✅ Test 7.1 PASSED: Complete workflow successful")
    return True
```

---

## 🧪 RUNNING ALL TESTS

### **Test Runner Script**

```python
# File: test_offline_suite.py

def run_all_tests():
    """Run all offline tests"""
    
    tests = [
        # Suite 1: Token Verification
        test_valid_token,
        test_tampered_token_hash,
        test_invalid_signature,
        test_expired_token,
        
        # Suite 2: Consent Checking
        test_module_allowed,
        test_module_blocked,
        
        # Suite 3: Extraction with Consent
        test_extract_allowed_module,
        test_extract_blocked_module,
        
        # Suite 4: Orchestration
        test_orchestrate_all_allowed,
        test_orchestrate_legal_level,
        test_orchestrate_standard_level,
        
        # Suite 5: Audit Logging
        test_extraction_logging,
        test_complete_audit_trail,
        
        # Suite 6: Error Handling
        test_missing_required_fields,
        
        # Suite 7: End-to-End
        test_complete_workflow
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("🧪 FORENSMART OFFLINE TESTING SUITE")
    print("=" * 70)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"⚠️ {test.__name__} ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
```

---

## 📋 TEST EXECUTION

### **Run Tests**

```bash
# Run all tests
python test_offline_suite.py

# Expected Output:
# ======================================================================
# 🧪 FORENSMART OFFLINE TESTING SUITE
# ======================================================================
# ✅ Test 1.1 PASSED: Valid token verified
# ✅ Test 1.2 PASSED: Tampered token detected (hash)
# ✅ Test 1.3 PASSED: Invalid signature detected
# ✅ Test 1.4 PASSED: Expired token detected
# ✅ Test 2.1 PASSED: Module allowed by consent
# ✅ Test 2.2 PASSED: Module blocked by consent
# ✅ Test 3.1 PASSED: Allowed module extracted
# ✅ Test 3.2 PASSED: Blocked module not extracted
# ✅ Test 4.1 PASSED: All modules extracted
# ✅ Test 4.2 PASSED: LEGAL level extraction correct
# ✅ Test 4.3 PASSED: STANDARD level extraction correct
# ✅ Test 5.1 PASSED: Extraction logging works
# ✅ Test 5.2 PASSED: Complete audit trail recorded
# ✅ Test 6.1 PASSED: Missing fields detected
# ✅ Test 7.1 PASSED: Complete workflow successful
# ======================================================================
# 📊 TEST RESULTS
# ======================================================================
# ✅ Passed: 15
# ❌ Failed: 0
# 📊 Total: 15
# 📈 Success Rate: 100.0%
# ======================================================================
```

---

## ✅ TESTING CHECKLIST

### **Token Verification Tests**
- [x] Valid token
- [x] Tampered hash
- [x] Invalid signature
- [x] Expired token
- [x] Missing fields

### **Consent Tests**
- [x] Module allowed
- [x] Module blocked

### **Extraction Tests**
- [x] Extract allowed module
- [x] Extract blocked module

### **Orchestration Tests**
- [x] Extract all (FULL)
- [x] Extract partial (LEGAL)
- [x] Extract minimal (STANDARD)

### **Logging Tests**
- [x] Extraction logging
- [x] Complete audit trail

### **Error Handling Tests**
- [x] Missing required fields

### **End-to-End Tests**
- [x] Complete workflow

---

## 🚀 STATUS

**Offline Testing**: ✅ **READY**

**No Internet Required**: ✅ **YES**

**Hash-Based Verification**: ✅ **USED**

**Test Coverage**: ✅ **COMPREHENSIVE**

**Expected Success Rate**: ✅ **100%**

---

**Status**: 🚀 **READY FOR TESTING**
