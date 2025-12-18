# CONSENT APPROVAL INTEGRATION PLAN

**Date**: December 1, 2025  
**Time**: 16:38 UTC+05:30  
**Status**: [READY FOR IMPLEMENTATION]

---

## 🎯 OBJECTIVE

Create a complete **Consent Approval System** with:
- ✅ Database schema for storing approvals
- ✅ API endpoints for managing approvals
- ✅ Streamlit UI for consent approval
- ✅ Integration with existing consent module
- ✅ Approval link generation and validation
- ✅ Audit trail and history tracking

---

## 📋 IMPLEMENTATION PLAN

### **PHASE 1: DATABASE SCHEMA** (30 minutes)
**File**: `modules/database/consent_approval_schema.py`

**Create Tables**:
```sql
-- Approval Links Table
CREATE TABLE approval_links (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    token VARCHAR(255) UNIQUE NOT NULL,
    nominee_email VARCHAR(255) NOT NULL,
    consent_level VARCHAR(50) NOT NULL,
    approval_method VARCHAR(50),
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Consent Approvals Table
CREATE TABLE consent_approvals (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    nominee_email VARCHAR(255) NOT NULL,
    approval_link_id INTEGER REFERENCES approval_links(id),
    consent_level VARCHAR(50) NOT NULL,
    approval_method VARCHAR(50),
    approved_at TIMESTAMP,
    approved_by VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Approval History Table
CREATE TABLE approval_history (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    approval_id INTEGER REFERENCES consent_approvals(id),
    action VARCHAR(100) NOT NULL,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_email VARCHAR(255),
    ip_address VARCHAR(50)
);
```

**Status**: ⏳ NOT STARTED

---

### **PHASE 2: DATABASE OPERATIONS** (30 minutes)
**File**: `modules/database/consent_operations.py`

**Implement Functions**:

```python
# Create approval link
def create_approval_link(case_id, nominee_email, consent_level, expires_in_hours=24):
    """Generate and store approval link"""
    # Generate unique token
    # Store in database
    # Return link URL

# Get approval link
def get_approval_link(token):
    """Retrieve approval link by token"""
    # Validate token
    # Check expiration
    # Return link data

# Approve consent
def approve_consent(token, approval_method, pin_code=None, pattern=None):
    """Record consent approval"""
    # Validate token
    # Verify approval method
    # Store approval
    # Log history

# Get approval status
def get_approval_status(case_id):
    """Get current approval status"""
    # Query database
    # Return status

# Log approval event
def log_approval_event(case_id, action, details, user_email=None):
    """Log approval event for audit trail"""
    # Store in approval_history
    # Include timestamp
```

**Status**: ⏳ NOT STARTED

---

### **PHASE 3: API ENDPOINTS** (45 minutes)
**File**: `modules/api/consent_approval_endpoints.py`

**Implement Endpoints**:

```python
# Endpoint 1: Generate Approval Link
POST /api/approvals/generate-link
{
    "case_id": "CASE_001",
    "nominee_email": "nominee@example.com",
    "consent_level": "STANDARD",
    "expires_in_hours": 24
}
Response:
{
    "approval_link": "https://localhost:8501/approve?token=abc123xyz",
    "token": "abc123xyz",
    "expires_at": "2025-12-02T16:38:00Z"
}

# Endpoint 2: Get Approval Link Details
GET /api/approvals/link/{token}
Response:
{
    "case_id": "CASE_001",
    "nominee_email": "nominee@example.com",
    "consent_level": "STANDARD",
    "approval_method": "PIN",
    "expires_at": "2025-12-02T16:38:00Z",
    "status": "pending"
}

# Endpoint 3: Approve Consent
POST /api/approvals/{case_id}/approve
{
    "token": "abc123xyz",
    "approval_method": "PIN",
    "pin_code": "1234",
    "nominee_email": "nominee@example.com"
}
Response:
{
    "status": "approved",
    "approved_at": "2025-12-01T16:38:00Z",
    "case_id": "CASE_001"
}

# Endpoint 4: Get Approval Status
GET /api/approvals/{case_id}/status
Response:
{
    "case_id": "CASE_001",
    "status": "approved",
    "approval_level": "STANDARD",
    "approved_at": "2025-12-01T16:38:00Z",
    "nominee_email": "nominee@example.com"
}

# Endpoint 5: Get Approval History
GET /api/approvals/{case_id}/history
Response:
[
    {
        "action": "link_generated",
        "timestamp": "2025-12-01T16:30:00Z",
        "details": "Approval link generated"
    },
    {
        "action": "link_accessed",
        "timestamp": "2025-12-01T16:35:00Z",
        "details": "Nominee accessed approval link"
    },
    {
        "action": "approved",
        "timestamp": "2025-12-01T16:38:00Z",
        "details": "Consent approved via PIN"
    }
]
```

**Status**: ⏳ NOT STARTED

---

### **PHASE 4: STREAMLIT UI** (1 hour)
**File**: `pages/08_consent_approval.py`

**Create New Page with Tabs**:

#### **Tab 1: Generate Approval Link**
- Input: Case ID, Nominee Email, Consent Level
- Button: "Generate Link"
- Output: Approval link URL (copyable)
- Display: QR code for link
- Option: Send via email/SMS

#### **Tab 2: Approve Consent (Nominee View)**
- Input: Approval token (from URL parameter)
- Display: Case details, consent level, approval method options
- Options: PIN, Pattern, Biometric (simulated)
- Button: "Approve"
- Confirmation: Success message with timestamp

#### **Tab 3: Approval Status**
- Input: Case ID
- Display: Current approval status
- Show: Approved by, timestamp, consent level
- Show: Approval history timeline
- Option: Revoke approval

#### **Tab 4: Approval History**
- Display: All approvals for all cases
- Filters: By case, by date, by status
- Show: Timeline of events
- Export: Download history as CSV/PDF

**Status**: ⏳ NOT STARTED

---

### **PHASE 5: INTEGRATION** (30 minutes)
**Files to Update**:
- `modules/consent/models.py`
- `app.py`
- `modules/shared/database.py`
- `modules/shared/api.py`

**Integration Points**:

1. **Database Integration**
   - Import consent operations
   - Use database functions in API endpoints
   - Store all approvals in database

2. **API Integration**
   - Create FastAPI router for consent endpoints
   - Include in main API
   - Add authentication/validation

3. **Streamlit Integration**
   - Create new page: `pages/08_consent_approval.py`
   - Add to sidebar navigation
   - Link to existing consent module
   - Use API endpoints for data

4. **Consent Module Integration**
   - Update `ConsentManager` to use database
   - Update `ApprovalLinkGenerator` to use API
   - Store all approvals in database
   - Log all events to history

**Status**: ⏳ NOT STARTED

---

## 📊 IMPLEMENTATION TIMELINE

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Database Schema | 30 min | ⏳ Pending |
| 2 | Database Operations | 30 min | ⏳ Pending |
| 3 | API Endpoints | 45 min | ⏳ Pending |
| 4 | Streamlit UI | 1 hour | ⏳ Pending |
| 5 | Integration | 30 min | ⏳ Pending |
| | **TOTAL** | **3.5 hours** | ⏳ Pending |

---

## 🎯 ENTRY POINT ARCHITECTURE

### **Current Architecture**
```
app.py (Main Streamlit App)
  ├── pages/01_device_selector.py
  ├── pages/02_extraction.py
  ├── pages/03_analysis.py
  ├── pages/04_consent.py
  ├── pages/05_case_management.py
  ├── pages/06_investigation.py
  ├── pages/07_reports.py
  └── [NEW] pages/08_consent_approval.py
```

### **New Entry Point: Consent Approval Page**
```
pages/08_consent_approval.py
  ├── Tab 1: Generate Approval Link
  │   ├── Input: Case ID, Nominee Email, Consent Level
  │   ├── API Call: POST /api/approvals/generate-link
  │   └── Output: Approval Link URL
  │
  ├── Tab 2: Approve Consent (Nominee View)
  │   ├── Input: Token from URL parameter
  │   ├── API Call: GET /api/approvals/link/{token}
  │   ├── User Action: Approve with PIN/Pattern/Biometric
  │   └── API Call: POST /api/approvals/{case_id}/approve
  │
  ├── Tab 3: Approval Status
  │   ├── Input: Case ID
  │   ├── API Call: GET /api/approvals/{case_id}/status
  │   └── Display: Current status and history
  │
  └── Tab 4: Approval History
      ├── Input: Case ID (optional)
      ├── API Call: GET /api/approvals/{case_id}/history
      └── Display: Timeline of all events
```

---

## 🔗 DATABASE INTEGRATION

### **Connection Flow**
```
Streamlit UI
    ↓
API Endpoints (FastAPI)
    ↓
Database Operations (SQLAlchemy)
    ↓
PostgreSQL Database
    ↓
Approval Links, Approvals, History Tables
```

### **Data Flow Example**
```
1. User clicks "Generate Link"
   ↓
2. Streamlit calls: POST /api/approvals/generate-link
   ↓
3. API calls: create_approval_link()
   ↓
4. Database stores: approval_links record
   ↓
5. API returns: approval link URL
   ↓
6. Streamlit displays: link with QR code
```

---

## ✅ IMPLEMENTATION CHECKLIST

### **Phase 1: Database Schema**
- [ ] Create `modules/database/consent_approval_schema.py`
- [ ] Define SQLAlchemy models for tables
- [ ] Create migration script
- [ ] Run migrations on PostgreSQL
- [ ] Verify tables created

### **Phase 2: Database Operations**
- [ ] Create `modules/database/consent_operations.py`
- [ ] Implement `create_approval_link()`
- [ ] Implement `get_approval_link()`
- [ ] Implement `approve_consent()`
- [ ] Implement `get_approval_status()`
- [ ] Implement `log_approval_event()`
- [ ] Test all functions

### **Phase 3: API Endpoints**
- [ ] Create `modules/api/consent_approval_endpoints.py`
- [ ] Implement POST /api/approvals/generate-link
- [ ] Implement GET /api/approvals/link/{token}
- [ ] Implement POST /api/approvals/{case_id}/approve
- [ ] Implement GET /api/approvals/{case_id}/status
- [ ] Implement GET /api/approvals/{case_id}/history
- [ ] Test all endpoints with Postman/curl

### **Phase 4: Streamlit UI**
- [ ] Create `pages/08_consent_approval.py`
- [ ] Implement Tab 1: Generate Link
- [ ] Implement Tab 2: Approve Consent
- [ ] Implement Tab 3: Approval Status
- [ ] Implement Tab 4: Approval History
- [ ] Add styling and formatting
- [ ] Test all UI components

### **Phase 5: Integration**
- [ ] Update `modules/consent/models.py`
- [ ] Update `app.py` (if needed)
- [ ] Update `modules/shared/database.py`
- [ ] Update `modules/shared/api.py`
- [ ] Test end-to-end flow
- [ ] Verify database persistence
- [ ] Verify API calls work

---

## 🚀 QUICK START AFTER IMPLEMENTATION

```bash
# 1. Start ForenSmart
cd c:\Forensmart
.\venv\Scripts\Activate.ps1
streamlit run app.py

# 2. Navigate to "Consent Approval" page in sidebar

# 3. Generate approval link
   - Enter Case ID
   - Enter Nominee Email
   - Select Consent Level
   - Click "Generate Link"

# 4. Share link with nominee
   - Copy link or QR code
   - Send via email/SMS

# 5. Nominee approves
   - Click link
   - Verify case details
   - Enter PIN/Pattern
   - Click "Approve"

# 6. Check status
   - Go to "Approval Status" tab
   - View approval details
   - See approval history
```

---

## 📝 NOTES

- All data persisted in PostgreSQL
- All approvals logged in history
- All events timestamped
- Approval links expire after 24 hours (configurable)
- PIN/Pattern/Biometric validation implemented
- Audit trail complete
- Ready for production

---

## ✅ SUMMARY

**Current Status**: Ready for implementation  
**Total Time**: 3.5 hours  
**Complexity**: Medium  
**Dependencies**: PostgreSQL, FastAPI, Streamlit  
**Result**: Complete consent approval system with database, API, and UI

---

**Next Step**: Start implementation with Phase 1 (Database Schema)

