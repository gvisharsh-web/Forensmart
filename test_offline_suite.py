"""
OFFLINE TESTING SUITE FOR FORENSMART
Complete testing without internet, real devices, or external dependencies

Tests:
- Token verification (hash + signature)
- Consent checking
- Extraction with consent
- Orchestration
- Audit logging
- Error handling
- End-to-end workflow
"""

import hashlib
import hmac
import json
import base64
from datetime import datetime, timedelta
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.extraction.consent_based_extraction import (
    ConsentTokenVerifier,
    DeviceInfoExtractor,
    CommunicationsExtractor,
    LocationExtractor,
    MediaExtractor,
    SecurityExtractor,
    SystemExtractor,
    ExtractionOrchestrator
)


# ============================================================================
# TEST SUITE 1: TOKEN VERIFICATION
# ============================================================================

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
    
    return True


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
    
    return True


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
    
    return True


def test_expired_token():
    """Test expired token detection"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 2: CONSENT CHECKING
# ============================================================================

def test_module_allowed():
    """Test module allowed by consent"""
    
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
    
    return True


def test_module_blocked():
    """Test module blocked by consent"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 3: EXTRACTION WITH CONSENT
# ============================================================================

def test_extract_allowed_module():
    """Test extraction of allowed module"""
    
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
    
    return True


def test_extract_blocked_module():
    """Test extraction of blocked module"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 4: ORCHESTRATION
# ============================================================================

def test_orchestrate_all_allowed():
    """Test orchestration with all modules allowed"""
    
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
    
    return True


def test_orchestrate_legal_level():
    """Test orchestration with LEGAL consent level"""
    
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
    
    return True


def test_orchestrate_standard_level():
    """Test orchestration with STANDARD consent level"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 5: AUDIT LOGGING
# ============================================================================

def test_extraction_logging():
    """Test extraction logging"""
    
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
    
    return True


def test_complete_audit_trail():
    """Test complete audit trail"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 6: ERROR HANDLING
# ============================================================================

def test_missing_required_fields():
    """Test missing required fields in token"""
    
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
    
    return True


# ============================================================================
# TEST SUITE 7: END-TO-END WORKFLOW
# ============================================================================

def test_complete_workflow():
    """Test complete end-to-end workflow"""
    
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
    
    return True


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all offline tests"""
    
    tests = [
        # Suite 1: Token Verification
        ('Test 1.1: Valid Token', test_valid_token),
        ('Test 1.2: Tampered Hash', test_tampered_token_hash),
        ('Test 1.3: Invalid Signature', test_invalid_signature),
        ('Test 1.4: Expired Token', test_expired_token),
        
        # Suite 2: Consent Checking
        ('Test 2.1: Module Allowed', test_module_allowed),
        ('Test 2.2: Module Blocked', test_module_blocked),
        
        # Suite 3: Extraction with Consent
        ('Test 3.1: Extract Allowed', test_extract_allowed_module),
        ('Test 3.2: Extract Blocked', test_extract_blocked_module),
        
        # Suite 4: Orchestration
        ('Test 4.1: All Modules', test_orchestrate_all_allowed),
        ('Test 4.2: LEGAL Level', test_orchestrate_legal_level),
        ('Test 4.3: STANDARD Level', test_orchestrate_standard_level),
        
        # Suite 5: Audit Logging
        ('Test 5.1: Extraction Logging', test_extraction_logging),
        ('Test 5.2: Complete Audit Trail', test_complete_audit_trail),
        
        # Suite 6: Error Handling
        ('Test 6.1: Missing Fields', test_missing_required_fields),
        
        # Suite 7: End-to-End
        ('Test 7.1: Complete Workflow', test_complete_workflow)
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("🧪 FORENSMART OFFLINE TESTING SUITE")
    print("=" * 70)
    print()
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"⚠️ {test_name} ERROR: {str(e)}")
            failed += 1
    
    print()
    print("=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    if passed + failed > 0:
        print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
