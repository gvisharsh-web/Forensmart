"""
Test consent approval API endpoints
"""

import os
import sys
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Load environment variables
load_dotenv()

# Add modules to path
modules_path = os.path.join(os.path.dirname(__file__), 'modules')
sys.path.insert(0, modules_path)

# Import router directly
import importlib.util
spec = importlib.util.spec_from_file_location("consent_approval_endpoints", 
    os.path.join(modules_path, 'api', 'consent_approval_endpoints.py'))
consent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consent_module)
router = consent_module.router

# Create FastAPI app
app = FastAPI()
app.include_router(router)

# Create test client
client = TestClient(app)

print("")
print("=" * 80)
print("TESTING CONSENT APPROVAL API ENDPOINTS")
print("=" * 80)
print("")

try:
    # Test 1: Generate approval link
    print("[TEST 1] POST /api/approvals/generate-link")
    response = client.post(
        "/api/approvals/generate-link",
        json={
            "case_id": "CASE_001",
            "nominee_email": "nominee@example.com",
            "consent_level": "STANDARD",
            "approval_method": "PIN",
            "expires_in_hours": 24
        }
    )
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    token = data['token']
    print(f"Response:")
    print(f"  - Token: {token}")
    print(f"  - Case ID: {data['case_id']}")
    print(f"  - Nominee: {data['nominee_email']}")
    print(f"  - Approval Link: {data['approval_link']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 2: Get approval link details
    print("[TEST 2] GET /api/approvals/link/{token}")
    response = client.get(f"/api/approvals/link/{token}")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Response:")
    print(f"  - Case ID: {data['case_id']}")
    print(f"  - Nominee: {data['nominee_email']}")
    print(f"  - Consent Level: {data['consent_level']}")
    print(f"  - Status: {data['status']}")
    print(f"  - Is Valid: {data['is_valid']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 3: Approve consent
    print("[TEST 3] POST /api/approvals/{case_id}/approve")
    response = client.post(
        "/api/approvals/CASE_001/approve",
        json={
            "token": token,
            "approval_method": "PIN",
            "nominee_email": "nominee@example.com",
            "pin_code": "1234"
        }
    )
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Response:")
    print(f"  - Status: {data['status']}")
    print(f"  - Case ID: {data['case_id']}")
    print(f"  - Consent Level: {data['consent_level']}")
    print(f"  - Approval Method: {data['approval_method']}")
    print(f"  - Approved At: {data['approved_at']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 4: Get approval status
    print("[TEST 4] GET /api/approvals/{case_id}/status")
    response = client.get("/api/approvals/CASE_001/status")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Response:")
    print(f"  - Case ID: {data['case_id']}")
    print(f"  - Status: {data['status']}")
    print(f"  - Consent Level: {data['consent_level']}")
    print(f"  - Nominee: {data['nominee_email']}")
    print(f"  - Approved At: {data['approved_at']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 5: Get approval history
    print("[TEST 5] GET /api/approvals/{case_id}/history")
    response = client.get("/api/approvals/CASE_001/history")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Response:")
    print(f"  - Case ID: {data['case_id']}")
    print(f"  - Events: {len(data['events'])}")
    for i, event in enumerate(data['events'], 1):
        print(f"    {i}. {event['action']}: {event['details']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 6: Error handling - Invalid token
    print("[TEST 6] GET /api/approvals/link/invalid-token (Error handling)")
    response = client.get("/api/approvals/link/invalid-token")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    print(f"Error: {response.json()['detail']}")
    print("[OK] Test passed!")
    print("")
    
    # Test 7: Error handling - Invalid approval
    print("[TEST 7] POST /api/approvals/CASE_999/approve (Error handling)")
    response = client.post(
        "/api/approvals/CASE_999/approve",
        json={
            "token": "invalid-token",
            "approval_method": "PIN",
            "nominee_email": "nominee@example.com",
            "pin_code": "1234"
        }
    )
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"Error: {response.json()['detail']}")
    print("[OK] Test passed!")
    print("")
    
    print("=" * 80)
    print("ALL API TESTS PASSED!")
    print("=" * 80)
    print("")
    print("Summary:")
    print("  - Generated approval link")
    print("  - Retrieved link details")
    print("  - Approved consent")
    print("  - Retrieved approval status")
    print("  - Retrieved approval history")
    print("  - Error handling working correctly")
    print("  - All API endpoints working correctly")
    print("")
    
except AssertionError as e:
    print(f"[ERROR] TEST FAILED: {str(e)}")
    print("")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] TEST FAILED: {str(e)}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)
