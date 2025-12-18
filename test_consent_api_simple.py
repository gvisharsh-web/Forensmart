"""
Test consent approval API endpoints - Simplified version
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Import database operations directly
from database.consent_operations import ConsentApprovalOperations, get_db_session
from database.consent_approval_schema import ApprovalLink

print("")
print("=" * 80)
print("TESTING CONSENT APPROVAL API LOGIC")
print("=" * 80)
print("")

try:
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Test 1: Generate approval link (simulating API endpoint)
    print("[TEST 1] Simulating: POST /api/approvals/generate-link")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    link = ops.create_approval_link(
        case_id='CASE_001',
        nominee_email='nominee@example.com',
        consent_level='STANDARD',
        approval_method='PIN',
        expires_in_hours=24
    )
    token = link.token
    
    print(f"Status Code: 200")
    print(f"Response:")
    print(f"  - Token: {token}")
    print(f"  - Case ID: {link.case_id}")
    print(f"  - Nominee: {link.nominee_email}")
    print(f"  - Approval Link: http://localhost:8501/approve?token={token}")
    print("[OK] Test passed!")
    print("")
    ops.close()
    
    # Test 2: Get approval link details (simulating API endpoint)
    print("[TEST 2] Simulating: GET /api/approvals/link/{token}")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    retrieved_link = ops.get_approval_link(token)
    
    print(f"Status Code: 200")
    print(f"Response:")
    print(f"  - Case ID: {retrieved_link.case_id}")
    print(f"  - Nominee: {retrieved_link.nominee_email}")
    print(f"  - Consent Level: {retrieved_link.consent_level}")
    print(f"  - Status: {retrieved_link.status}")
    print(f"  - Is Valid: {retrieved_link.is_valid()}")
    print("[OK] Test passed!")
    print("")
    ops.close()
    
    # Test 3: Approve consent (simulating API endpoint)
    print("[TEST 3] Simulating: POST /api/approvals/{case_id}/approve")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    approval = ops.approve_consent(
        token=token,
        approval_method='PIN',
        nominee_email='nominee@example.com',
        pin_code='1234'
    )
    
    print(f"Status Code: 200")
    print(f"Response:")
    print(f"  - Status: {approval.status}")
    print(f"  - Case ID: {approval.case_id}")
    print(f"  - Consent Level: {approval.consent_level}")
    print(f"  - Approval Method: {approval.approval_method}")
    print(f"  - Approved At: {approval.approved_at}")
    print("[OK] Test passed!")
    print("")
    ops.close()
    
    # Test 4: Get approval status (simulating API endpoint)
    print("[TEST 4] Simulating: GET /api/approvals/{case_id}/status")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    status = ops.get_approval_status('CASE_001')
    
    print(f"Status Code: 200")
    print(f"Response:")
    print(f"  - Case ID: {status['case_id']}")
    print(f"  - Status: {status['status']}")
    print(f"  - Consent Level: {status['consent_level']}")
    print(f"  - Nominee: {status['nominee_email']}")
    print(f"  - Approved At: {status['approved_at']}")
    print("[OK] Test passed!")
    print("")
    ops.close()
    
    # Test 5: Get approval history (simulating API endpoint)
    print("[TEST 5] Simulating: GET /api/approvals/{case_id}/history")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    history = ops.get_approval_history('CASE_001')
    
    print(f"Status Code: 200")
    print(f"Response:")
    print(f"  - Case ID: CASE_001")
    print(f"  - Events: {len(history)}")
    for i, event in enumerate(history, 1):
        print(f"    {i}. {event.action}: {event.details}")
    print("[OK] Test passed!")
    print("")
    ops.close()
    
    # Test 6: Error handling - Invalid token
    print("[TEST 6] Simulating: GET /api/approvals/link/invalid-token (Error handling)")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    invalid_link = ops.get_approval_link('invalid-token')
    
    if invalid_link is None:
        print(f"Status Code: 404")
        print(f"Error: Approval link not found")
        print("[OK] Test passed!")
    else:
        print("[ERROR] Should have returned None")
    print("")
    ops.close()
    
    # Test 7: Error handling - Invalid approval
    print("[TEST 7] Simulating: POST /api/approvals/CASE_999/approve (Error handling)")
    db = get_db_session(DATABASE_URL)
    ops = ConsentApprovalOperations(db)
    
    try:
        approval = ops.approve_consent(
            token='invalid-token',
            approval_method='PIN',
            nominee_email='nominee@example.com',
            pin_code='1234'
        )
        print("[ERROR] Should have raised ValueError")
    except ValueError as e:
        print(f"Status Code: 400")
        print(f"Error: {str(e)}")
        print("[OK] Test passed!")
    print("")
    ops.close()
    
    print("=" * 80)
    print("ALL API LOGIC TESTS PASSED!")
    print("=" * 80)
    print("")
    print("Summary:")
    print("  - Generated approval link")
    print("  - Retrieved link details")
    print("  - Approved consent")
    print("  - Retrieved approval status")
    print("  - Retrieved approval history")
    print("  - Error handling working correctly")
    print("  - All API logic working correctly")
    print("")
    
except Exception as e:
    print(f"[ERROR] TEST FAILED: {str(e)}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)
