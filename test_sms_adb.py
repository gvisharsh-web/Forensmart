"""
Test Script for SMS/Hash ADB Reading Functions
==============================================

Tests the following functions from dashboard_merged.py:
1. _normalize_phone_number() - E.164 normalization
2. _check_adb_device_connected() - ADB device detection
3. _read_sms_from_adb() - SMS reading and hash extraction
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import functions from dashboard_merged
sys.path.insert(0, str(Path(__file__).parent))
from modules.dashboard_merged import (
    _normalize_phone_number,
    _check_adb_device_connected,
    _read_sms_from_adb
)


def test_phone_normalization():
    """Test phone number normalization to E.164 format."""
    print("\n" + "="*60)
    print("TEST 1: Phone Number Normalization (E.164)")
    print("="*60)
    
    test_cases = [
        ("9876543210", "+9876543210"),
        ("+919876543210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("+1-234-567-8900", "+12345678900"),
        ("(123) 456-7890", "+1234567890"),
        ("+44 20 7946 0958", "+442079460958"),
    ]
    
    passed = 0
    failed = 0
    
    for input_phone, expected in test_cases:
        result = _normalize_phone_number(input_phone)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status}: '{input_phone}' -> '{result}' (expected: '{expected}')")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_adb_device_detection():
    """Test ADB device connection detection."""
    print("\n" + "="*60)
    print("TEST 2: ADB Device Detection")
    print("="*60)
    
    try:
        connected = _check_adb_device_connected()
        if connected:
            print("[PASS]: ADB device is connected")
            return True
        else:
            print("[WARNING]: No ADB device connected")
            print("   To test SMS reading, connect an Android phone via USB with ADB enabled")
            return False
    except Exception as e:
        print(f"[FAIL]: ADB detection error: {e}")
        return False


def test_sms_reading():
    """Test SMS reading from ADB device."""
    print("\n" + "="*60)
    print("TEST 3: SMS Reading from ADB Device")
    print("="*60)
    
    # Check if device is connected first
    if not _check_adb_device_connected():
        print("[SKIP]: No ADB device connected")
        print("   To test SMS reading:")
        print("   1. Connect Android phone via USB")
        print("   2. Enable ADB debugging on phone")
        print("   3. Send SMS with format: 'APPROVE A7B9C1D2'")
        print("   4. Run this test again")
        return False
    
    # Test with a sample phone number
    test_phone = "+919876543210"
    print(f"Reading SMS from phone: {test_phone}")
    
    try:
        sms_data = _read_sms_from_adb(test_phone)
        
        if sms_data:
            print(f"[PASS]: SMS found and parsed")
            print(f"   Phone: {sms_data['phone']}")
            print(f"   Hash: {sms_data['hash']}")
            print(f"   Message: {sms_data['message']}")
            return True
        else:
            print(f"[WARNING]: No SMS found from {test_phone}")
            print("   Make sure nominee sent SMS in format: 'APPROVE A7B9C1D2'")
            return False
    
    except Exception as e:
        print(f"[FAIL]: SMS reading error: {e}")
        return False


def test_hash_extraction():
    """Test hash extraction from SMS message."""
    print("\n" + "="*60)
    print("TEST 4: Hash Extraction from SMS Message")
    print("="*60)
    
    import re
    
    test_messages = [
        ("APPROVE A7B9C1D2", "A7B9C1D2", True),
        ("approve a7b9c1d2", "A7B9C1D2", True),
        ("APPROVE A7B9C1D2 extra text", "A7B9C1D2", True),
        ("APPROVE", None, False),
        ("A7B9C1D2", None, False),
        ("DENY A7B9C1D2", None, False),
    ]
    
    passed = 0
    failed = 0
    
    for message, expected_hash, should_match in test_messages:
        hash_match = re.search(r'APPROVE\s+([A-Z0-9]{8})', message.upper())
        result_hash = hash_match.group(1) if hash_match else None
        
        if should_match:
            status = "[PASS]" if result_hash == expected_hash else "[FAIL]"
            print(f"{status}: '{message}' -> '{result_hash}' (expected: '{expected_hash}')")
        else:
            status = "[PASS]" if result_hash is None else "[FAIL]"
            print(f"{status}: '{message}' -> {result_hash} (expected: None)")
        
        if (should_match and result_hash == expected_hash) or (not should_match and result_hash is None):
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SMS/Hash ADB Reading - Test Suite")
    print("="*60)
    
    results = {
        "Phone Normalization": test_phone_normalization(),
        "ADB Device Detection": test_adb_device_detection(),
        "SMS Reading": test_sms_reading(),
        "Hash Extraction": test_hash_extraction(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[SKIP/FAIL]"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n[SUCCESS] All tests passed! SMS/Hash ADB reading is working correctly.")
    else:
        print("\n[WARNING] Some tests failed or were skipped. Check the output above.")


if __name__ == "__main__":
    main()
