# DATABASE & API FOR CONSENT APPROVAL - COMPLETE STATUS REPORT

**Date**: December 1, 2025  
**Status**: ✅ PARTIALLY COMPLETE - NEEDS ENHANCEMENT FOR CONSENT APPROVAL

---

## 🎯 EXECUTIVE SUMMARY

**Current Status**:
- ✅ Database module exists (`modules/shared/database.py`)
- ✅ API module exists (`modules/shared/api.py`)
- ✅ Database tools available (SQLAlchemy, PostgreSQL, Redis)
- ❌ **NOT INTEGRATED** with consent approval system
- ❌ **NO CONSENT APPROVAL ENDPOINTS** in API
- ❌ **NO CONSENT APPROVAL TABLES** in database schema

---

## 📊 CURRENT DATABASE MODULE

**File**: `modules/shared/database.py` (274 lines)

### What Exists
```python
class DatabaseManager:
    - connect()          # Connect to database
    - disconnect()       # Disconnect
    - is_connected()     # Check connection
    - create()          # Create record
    - read()            # Read records
    - update()          # Update record
    - delete()          # Delete record
    - query()           # Execute query
```

### What's Missing for Consent Approval
- ❌ No consent approval table schema
- ❌ No approval link storage
- ❌ No approval history tracking
- ❌ No nominee information storage
- ❌ No case-approval relationship
- ❌ No sync mechanism for online/offline

---

## 📡 CURRENT API MODULE

**File**: `modules/shared/api.py` (241 lines)

### What Exists
```python
class APIClient:
    - register_endpoint()    # Register endpoint
    - get_endpoint()         # Get endpoint details
    - list_endpoints()       # List all endpoints
    - get()                 # GET request
    - post()                # POST request
    - put()                 # PUT request
    - delete()              # DELETE request
```

### What's Missing for Consent Approval
- ❌ No consent approval endpoints
- ❌ No approval link endpoints
- ❌ No nominee approval endpoints
- ❌ No approval status endpoints
- ❌ No sync endpoints
- ❌ No webhook support

---

## 🔴 CRITICAL GAPS

### Gap 1: No Consent Approval Database Schema

**Missing Tables**:
```sql
-- Approval Links Table
CREATE TABLE approval_links (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    approved BOOLEAN DEFAULT FALSE,
    approval_time TIMESTAMP,
    approval_method VARCHAR(20),
    nominee_name VARCHAR(100),
    nominee_email VARCHAR(100),
    nominee_phone VARCHAR(20)
);

-- Consent Approvals Table
CREATE TABLE consent_approvals (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    approval_link_id INT REFERENCES approval_links(id),
    consent_level VARCHAR(20),
    approved_by VARCHAR(100),
    approval_method VARCHAR(20),
    pin_hash VARCHAR(255),
    approved_at TIMESTAMP,
    investigator_notified BOOLEAN DEFAULT FALSE,
    notified_at TIMESTAMP
);

-- Approval History Table
CREATE TABLE approval_history (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    event VARCHAR(50),
    actor VARCHAR(100),
    actor_role VARCHAR(20),
    timestamp TIMESTAMP,
    details JSON
);
```

---

### Gap 2: No API Endpoints for Consent Approval

**Missing Endpoints**:
```python
# Approval Link Management
POST   /api/approvals/generate-link         # Generate approval link
GET    /api/approvals/link/{token}          # Get link details
PUT    /api/approvals/link/{token}          # Update link
DELETE /api/approvals/link/{token}          # Delete link

# Approval Processing
POST   /api/approvals/{case_id}/approve     # Submit approval
GET    /api/approvals/{case_id}/status      # Get approval status
POST   /api/approvals/{case_id}/verify-pin  # Verify PIN
POST   /api/approvals/{case_id}/verify-pattern  # Verify pattern

# Approval Notifications
POST   /api/approvals/{case_id}/notify      # Notify investigator
GET    /api/approvals/{case_id}/notifications  # Get notifications

# Approval History
GET    /api/approvals/{case_id}/history     # Get approval history
GET    /api/approvals/audit-trail           # Get audit trail
```

---

### Gap 3: No Integration Between Modules

**Missing Connections**:
- ❌ Database module doesn't know about consent approvals
- ❌ API module doesn't call database for approvals
- ❌ Consent module doesn't use database/API
- ❌ No data synchronization
- ❌ No error handling between layers

---

## 🔧 REQUIRED IMPLEMENTATION

### Phase 1: Database Schema for Consent Approval (1 hour)

**File to Create**: `modules/database/consent_approval_schema.py`

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ApprovalLink(Base):
    __tablename__ = 'approval_links'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    approved = Column(Boolean, default=False)
    approval_time = Column(DateTime)
    approval_method = Column(String(20))
    nominee_name = Column(String(100))
    nominee_email = Column(String(100))
    nominee_phone = Column(String(20))

class ConsentApproval(Base):
    __tablename__ = 'consent_approvals'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), nullable=False)
    approval_link_id = Column(Integer, ForeignKey('approval_links.id'))
    consent_level = Column(String(20), nullable=False)
    approved_by = Column(String(100), nullable=False)
    approval_method = Column(String(20), nullable=False)
    pin_hash = Column(String(255))
    approved_at = Column(DateTime, nullable=False)
    investigator_notified = Column(Boolean, default=False)
    notified_at = Column(DateTime)

class ApprovalHistory(Base):
    __tablename__ = 'approval_history'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(50), nullable=False)
    event = Column(String(50), nullable=False)
    actor = Column(String(100), nullable=False)
    actor_role = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    details = Column(JSON)
```

---

### Phase 2: Database Operations for Consent (1 hour)

**File to Create**: `modules/database/consent_operations.py`

```python
from sqlalchemy.orm import Session
from modules.database.consent_approval_schema import ApprovalLink, ConsentApproval, ApprovalHistory
from datetime import datetime, timedelta

class ConsentApprovalDB:
    def __init__(self, db_session: Session):
        self.session = db_session
    
    def create_approval_link(self, case_id: str, token: str, expiry_hours: int = 1) -> ApprovalLink:
        """Create approval link"""
        link = ApprovalLink(
            case_id=case_id,
            token=token,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=expiry_hours)
        )
        self.session.add(link)
        self.session.commit()
        return link
    
    def get_approval_link(self, token: str) -> ApprovalLink:
        """Get approval link by token"""
        return self.session.query(ApprovalLink).filter_by(token=token).first()
    
    def approve_consent(self, case_id: str, consent_level: str, approved_by: str, 
                       approval_method: str, pin_hash: str = None) -> ConsentApproval:
        """Record consent approval"""
        approval = ConsentApproval(
            case_id=case_id,
            consent_level=consent_level,
            approved_by=approved_by,
            approval_method=approval_method,
            pin_hash=pin_hash,
            approved_at=datetime.now()
        )
        self.session.add(approval)
        self.session.commit()
        return approval
    
    def get_approval_status(self, case_id: str) -> ConsentApproval:
        """Get approval status for case"""
        return self.session.query(ConsentApproval).filter_by(case_id=case_id).first()
    
    def log_approval_event(self, case_id: str, event: str, actor: str, 
                          actor_role: str, details: dict = None) -> ApprovalHistory:
        """Log approval event"""
        history = ApprovalHistory(
            case_id=case_id,
            event=event,
            actor=actor,
            actor_role=actor_role,
            timestamp=datetime.now(),
            details=details
        )
        self.session.add(history)
        self.session.commit()
        return history
```

---

### Phase 3: API Endpoints for Consent Approval (1.5 hours)

**File to Create**: `modules/api/consent_approval_endpoints.py`

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from modules.database.consent_operations import ConsentApprovalDB

router = APIRouter(prefix="/api/approvals", tags=["approvals"])

class ApprovalLinkRequest(BaseModel):
    case_id: str
    expiry_hours: int = 1
    nominee_name: str
    nominee_email: str

class ApprovalRequest(BaseModel):
    case_id: str
    consent_level: str
    approval_method: str
    pin_hash: str = None

@router.post("/generate-link")
async def generate_approval_link(request: ApprovalLinkRequest, db: ConsentApprovalDB):
    """Generate approval link"""
    try:
        token = secrets.token_urlsafe(32)
        link = db.create_approval_link(request.case_id, token, request.expiry_hours)
        
        approval_url = f"http://localhost:8501/app?approve={request.case_id}&token={token}"
        
        return {
            'success': True,
            'token': token,
            'approval_url': approval_url,
            'expires_at': link.expires_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/link/{token}")
async def get_link_details(token: str, db: ConsentApprovalDB):
    """Get approval link details"""
    link = db.get_approval_link(token)
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    
    return {
        'case_id': link.case_id,
        'created_at': link.created_at.isoformat(),
        'expires_at': link.expires_at.isoformat(),
        'approved': link.approved,
        'approval_time': link.approval_time.isoformat() if link.approval_time else None
    }

@router.post("/{case_id}/approve")
async def approve_consent(case_id: str, request: ApprovalRequest, db: ConsentApprovalDB):
    """Submit consent approval"""
    try:
        approval = db.approve_consent(
            case_id=case_id,
            consent_level=request.consent_level,
            approved_by=request.approval_method,
            approval_method=request.approval_method,
            pin_hash=request.pin_hash
        )
        
        # Log event
        db.log_approval_event(
            case_id=case_id,
            event='APPROVAL_SUBMITTED',
            actor='NOMINEE',
            actor_role='NOMINEE',
            details={'method': request.approval_method}
        )
        
        return {
            'success': True,
            'case_id': case_id,
            'approved_at': approval.approved_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{case_id}/status")
async def get_approval_status(case_id: str, db: ConsentApprovalDB):
    """Get approval status"""
    approval = db.get_approval_status(case_id)
    
    if not approval:
        return {
            'case_id': case_id,
            'status': 'PENDING',
            'approved': False
        }
    
    return {
        'case_id': case_id,
        'status': 'APPROVED',
        'approved': True,
        'consent_level': approval.consent_level,
        'approved_at': approval.approved_at.isoformat(),
        'approval_method': approval.approval_method
    }
```

---

### Phase 4: Integration with Consent Module (1 hour)

**Update**: `modules/consent/models.py`

```python
from modules.database.consent_operations import ConsentApprovalDB

class ConsentManager:
    def __init__(self, storage_path: str = "consent_records", db_session = None):
        self.storage_path = storage_path
        self.db_session = db_session
        self.db = ConsentApprovalDB(db_session) if db_session else None
    
    def create_session(self, case_id: str, level: ConsentLevel, approved_by: str, 
                      approval_method: str) -> ConsentSession:
        """Create consent session with database persistence"""
        
        # Create in-memory session
        session = ConsentSession(
            case_id=case_id,
            level=level,
            approved_by=approved_by,
            approval_method=approval_method,
            timestamp=datetime.now()
        )
        
        # Save to database if available
        if self.db:
            self.db.approve_consent(
                case_id=case_id,
                consent_level=level.name,
                approved_by=approved_by,
                approval_method=approval_method
            )
        
        # Save to file (fallback)
        self._save_session(session)
        
        return session
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Database Schema
- [ ] Create consent_approval_schema.py
- [ ] Define ApprovalLink table
- [ ] Define ConsentApproval table
- [ ] Define ApprovalHistory table
- [ ] Create database migrations
- [ ] Test schema creation

### Database Operations
- [ ] Create consent_operations.py
- [ ] Implement create_approval_link()
- [ ] Implement get_approval_link()
- [ ] Implement approve_consent()
- [ ] Implement get_approval_status()
- [ ] Implement log_approval_event()
- [ ] Test all operations

### API Endpoints
- [ ] Create consent_approval_endpoints.py
- [ ] Implement POST /generate-link
- [ ] Implement GET /link/{token}
- [ ] Implement POST /{case_id}/approve
- [ ] Implement GET /{case_id}/status
- [ ] Add error handling
- [ ] Test all endpoints

### Integration
- [ ] Update ConsentManager to use database
- [ ] Update ApprovalLinkGenerator to use database
- [ ] Update app.py to use API endpoints
- [ ] Add database connection initialization
- [ ] Test end-to-end flow

### Testing
- [ ] Unit tests for database operations
- [ ] Unit tests for API endpoints
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Performance tests

---

## 🎯 TIMELINE

| Task | Duration | Status |
|------|----------|--------|
| Database Schema | 1 hour | Pending |
| Database Operations | 1 hour | Pending |
| API Endpoints | 1.5 hours | Pending |
| Integration | 1 hour | Pending |
| Testing | 1.5 hours | Pending |
| **TOTAL** | **6 hours** | **Pending** |

---

## 🚀 PRIORITY

**Priority**: HIGH  
**Blocking**: YES (Consent approval workflow broken without this)  
**Impact**: Cannot share approval data between investigator and nominee

---

## 📝 SUMMARY

**Current Status**:
- ✅ Database module exists (basic)
- ✅ API module exists (basic)
- ❌ **NOT INTEGRATED** with consent approval
- ❌ **NO CONSENT APPROVAL SCHEMA**
- ❌ **NO CONSENT APPROVAL ENDPOINTS**

**What's Needed**:
1. Consent approval database schema
2. Database operations for consent
3. API endpoints for consent approval
4. Integration with existing consent module
5. Testing and validation

**Estimated Effort**: 6 hours

---

**Status**: ✅ ANALYSIS COMPLETE - READY FOR IMPLEMENTATION

