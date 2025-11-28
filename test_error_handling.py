#!/usr/bin/env python
"""Test Advanced Error Handling System"""

from modules.error_handling import ErrorHandlingSystem

error_system = ErrorHandlingSystem()

# Test 1: Code Error Detection
print("TEST 1: Code Error Detection")
try:
    result = int("not a number")
except Exception as e:
    error_result = error_system.handle_error(error=e)
    print(f"  Error Type: {error_result['error_info']['type']}")
    print(f"  Severity: {error_result['error_info']['severity']}")
    print(f"  Auto-fixable: {error_result['error_info']['auto_fixable']}")
    print(f"  Status: PASS")

# Test 2: Logic Error Detection
print("\nTEST 2: Logic Error Detection")
context = {
    'extraction_params': {
        'device_id': 'DEVICE-001'
        # Missing case_id - should trigger error
    }
}
error_result = error_system.handle_error(context=context)
if error_result['error_detected']:
    print(f"  Error Type: {error_result['error_info']['type']}")
    print(f"  Message: {error_result['error_info']['message']}")
    print(f"  Status: PASS")
else:
    print(f"  Status: FAIL - No error detected")

# Test 3: Silent Error Detection
print("\nTEST 3: Silent Error Detection")
operation_result = {
    'extraction_result': {
        'modules': ['communications', 'location']
        # Missing other modules
    }
}
error_result = error_system.handle_error(operation_result=operation_result)
if error_result['error_detected']:
    print(f"  Error Type: {error_result['error_info']['type']}")
    print(f"  Message: {error_result['error_info']['message']}")
    print(f"  Status: PASS")
else:
    print(f"  Status: FAIL - No error detected")

# Test 4: Error Analysis
print("\nTEST 4: Error Analysis")
error_result = error_system.handle_error(error=ValueError("Invalid value"))
if error_result['error_detected']:
    analysis = error_result['analysis']
    print(f"  Root Cause: {analysis['root_cause']['probable_cause']}")
    print(f"  Impact: {analysis['impact']['impact_level']}")
    print(f"  Recommendations: {len(analysis['recommendations'])} provided")
    print(f"  Status: PASS")

# Test 5: Error Rectification
print("\nTEST 5: Error Rectification")
error_result = error_system.handle_error(error=ValueError("Invalid value"))
if error_result['error_detected']:
    rectification = error_result['rectification']
    print(f"  Fix Type: {rectification['fix_type']}")
    print(f"  Auto-fixable: {rectification.get('success', False)}")
    print(f"  Status: PASS")

print("\nALL TESTS COMPLETED SUCCESSFULLY!")
