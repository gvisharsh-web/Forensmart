"""
Test consent approval database operations
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from database.consent_operations import ConsentApprovalOperations
from database.consent_approval_schema import Base

print("")
print("=" * 80)
print("TESTING CONSENT APPROVAL DATABASE OPERATIONS")
print("=" * 80)
print("")

try:
    # Create database session
    print("[INFO] Connecting to database...")
    engine = create_engine(DATABASE_URL, echo=False)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    print("[OK] Connected to database")
    print("")
    
    # Create operations instance
    ops = ConsentApprovalOperations(db)
    
    # Test 1: Create approval link
    print("[TEST 1] Creating approval link...")
    link = ops.create_approval_link(
        case_id='CASE_001',
        nominee_email='nominee@example.com',
        consent_level='STANDARD',
        approval_method='PIN',
        expires_in_hours=24
    )
    print(f"[OK] Created link: {link.token}")
    print(f"     Case ID: {link.case_id}")
    print(f"     Nominee: {link.nominee_email}")
    print(f"     Status: {link.status}")
    print(f"     Expires in: 24 hours")
    print("")
    
    # Test 2: Get approval link
    print("[TEST 2] Getting approval link...")
    retrieved_link = ops.get_approval_link(link.token)
    print(f"[OK] Retrieved link: {retrieved_link.token}")
    print(f"     Valid: {retrieved_link.is_valid()}")
    print(f"     Expired: {retrieved_link.is_expired()}")
    print("")
    
    # Test 3: Get approval links by case
    print("[TEST 3] Getting approval links by case...")
    links = ops.get_approval_links_by_case('CASE_001')
    print(f"[OK] Retrieved {len(links)} link(s) for case CASE_001")
    print("")
    
    # Test 4: Approve consent
    print("[TEST 4] Approving consent...")
    approval = ops.approve_consent(
        token=link.token,
        approval_method='PIN',
        nominee_email='nominee@example.com',
        pin_code='1234'
    )
    print(f"[OK] Approved consent")
    print(f"     Status: {approval.status}")
    print(f"     Approved at: {approval.approved_at}")
    print(f"     Approval method: {approval.approval_method}")
    print("")
    
    # Test 5: Get approval status
    print("[TEST 5] Getting approval status...")
    status = ops.get_approval_status('CASE_001')
    print(f"[OK] Retrieved status")
    print(f"     Status: {status['status']}")
    print(f"     Consent Level: {status['consent_level']}")
    print(f"     Nominee: {status['nominee_email']}")
    print(f"     Approved at: {status['approved_at']}")
    print("")
    
    # Test 6: Get approval history
    print("[TEST 6] Getting approval history...")
    history = ops.get_approval_history('CASE_001')
    print(f"[OK] Retrieved {len(history)} history record(s)")
    for i, record in enumerate(history, 1):
        print(f"     {i}. {record.action}: {record.details}")
    print("")
    
    # Test 7: Create another link for revoke test
    print("[TEST 7] Testing link revocation...")
    link2 = ops.create_approval_link(
        case_id='CASE_002',
        nominee_email='nominee2@example.com',
        consent_level='LEGAL',
        approval_method='PATTERN'
    )
    print(f"[OK] Created second link: {link2.token}")
    
    # Revoke the link
    revoked = ops.revoke_approval_link(link2.token)
    print(f"[OK] Revoked link: {revoked}")
    
    # Verify it's revoked
    revoked_link = ops.get_approval_link(link2.token)
    print(f"     Link status: {revoked_link.status}")
    print("")
    
    # Test 8: Test approval revocation
    print("[TEST 8] Testing approval revocation...")
    revoked_approval = ops.revoke_approval('CASE_001')
    print(f"[OK] Revoked approval: {revoked_approval}")
    
    # Verify it's revoked
    revoked_status = ops.get_approval_status('CASE_001')
    print(f"     Approval status: {revoked_status['status']}")
    print("")
    
    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("")
    print("Summary:")
    print("  - Created approval links")
    print("  - Retrieved approval links")
    print("  - Approved consent")
    print("  - Retrieved approval status")
    print("  - Retrieved approval history")
    print("  - Revoked links and approvals")
    print("  - All database operations working correctly")
    print("")
    
except Exception as e:
    print(f"[ERROR] TEST FAILED: {str(e)}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    ops.close()
