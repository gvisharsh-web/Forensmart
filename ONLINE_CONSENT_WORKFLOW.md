# 🔌 ONLINE CONSENT WORKFLOW - COMPLETE

**Status**: Online consent workflow with artifact routing
**Date**: November 25, 2025

---

## ✅ ONLINE CONSENT WORKFLOW

### **SCENARIO: User is ONLINE**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ONLINE CONSENT WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: USER INITIATES CONSENT (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ Streamlit UI (modules/consent/ui.py)                                    │
├──────────────────────────────────────────────────────────────────────────┤
│ • User enters case ID: "CASE-2025-001"                                  │
│ • User enters consent level: "FULL"                                     │
│ • User enters nominee details (name, email, phone)                      │
│ • User clicks: "Request Consent"                                        │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 2: CREATE CONSENT SESSION (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ConsentManager (modules/consent/models.py)                              │
├──────────────────────────────────────────────────────────────────────────┤
│ • create_consent_session()                                              │
│   ├─ Case ID: "CASE-2025-001"                                           │
│   ├─ Consent Level: "FULL"                                              │
│   ├─ Nominee: {name, email, phone}                                      │
│   ├─ Session ID: Generated (unique)                                     │
│   ├─ Status: "PENDING"                                                  │
│   ├─ Created At: 2025-11-25 20:51:00                                    │
│   └─ Expires At: 2025-11-26 20:51:00 (24 hours)                         │
│                                                                          │
│ HybridConnectivityManager:                                              │
│ • is_online = True ✅                                                   │
│ • pending_sync_queue = []                                               │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 3: SAVE CONSENT SESSION TO ARTIFACTS (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ArtifactPathBuilder + ResultsRepository                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ • Resolve path:                                                         │
│   artifacts/CASE-2025-001/consent/                                      │
│                                                                          │
│ • save_consent_session(case_id, session)                                │
│   ├─ Create directory: artifacts/CASE-2025-001/consent/                 │
│   ├─ Save to: sessions.json                                             │
│   └─ Content:                                                           │
│       {                                                                 │
│         "case_id": "CASE-2025-001",                                     │
│         "session_id": "sess_abc123xyz",                                 │
│         "consent_level": "FULL",                                        │
│         "nominee": {...},                                               │
│         "status": "PENDING",                                            │
│         "created_at": "2025-11-25T20:51:00"                             │
│       }                                                                 │
│                                                                          │
│ • Also save to ResultsRepository:                                       │
│   └─ results.json with consent_session key                              │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 4: GENERATE APPROVAL LINK (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ApprovalLinkGenerator (modules/consent/models.py)                       │
├──────────────────────────────────────────────────────────────────────────┤
│ • generate_approval_link()                                              │
│   ├─ Session ID: "sess_abc123xyz"                                       │
│   ├─ Token: Generated (secure, one-time use)                            │
│   ├─ Link: "https://forensmart.app/consent/approve?token=xyz789"        │
│   └─ Expires: 24 hours                                                  │
│                                                                          │
│ • Send approval link to nominee:                                        │
│   ├─ Email: "nominee@example.com"                                       │
│   ├─ SMS: "+1-555-0123" (optional)                                      │
│   └─ Message: "Please approve consent for case CASE-2025-001"           │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 5: NOMINEE APPROVES (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ Nominee receives link and clicks approval                               │
├──────────────────────────────────────────────────────────────────────────┤
│ • Nominee clicks: "Approve Consent"                                     │
│ • Token validated: Valid ✅                                             │
│ • Session found: "sess_abc123xyz" ✅                                    │
│ • Approval recorded:                                                    │
│   ├─ Approved At: 2025-11-25 20:55:00                                   │
│   ├─ Approved By: Nominee name                                          │
│   ├─ IP Address: 192.168.1.100                                          │
│   └─ User Agent: Mozilla/5.0...                                         │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 6: SAVE APPROVAL RECORD TO ARTIFACTS (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ArtifactPathBuilder + ResultsRepository                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ • save_approval_record(case_id, approval)                               │
│   ├─ Path: artifacts/CASE-2025-001/consent/                             │
│   ├─ Save to: approvals.json                                            │
│   └─ Content (append):                                                  │
│       [                                                                 │
│         {                                                               │
│           "session_id": "sess_abc123xyz",                               │
│           "approved_at": "2025-11-25T20:55:00",                         │
│           "approved_by": "Nominee Name",                                │
│           "ip_address": "192.168.1.100",                                │
│           "status": "APPROVED"                                          │
│         }                                                               │
│       ]                                                                 │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 7: UPDATE CONSENT SESSION STATUS (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ConsentManager (modules/consent/models.py)                              │
├──────────────────────────────────────────────────────────────────────────┤
│ • approve_consent()                                                     │
│   ├─ Session Status: "PENDING" → "APPROVED"                             │
│   ├─ Approved At: 2025-11-25 20:55:00                                   │
│   ├─ Approval Token: Marked as used                                     │
│   └─ Audit Trail: Updated                                               │
│                                                                          │
│ • save_consent_session() - UPDATE                                       │
│   ├─ Path: artifacts/CASE-2025-001/consent/sessions.json                │
│   └─ Updated Content:                                                   │
│       {                                                                 │
│         "case_id": "CASE-2025-001",                                     │
│         "session_id": "sess_abc123xyz",                                 │
│         "consent_level": "FULL",                                        │
│         "nominee": {...},                                               │
│         "status": "APPROVED",                                           │
│         "created_at": "2025-11-25T20:51:00",                            │
│         "approved_at": "2025-11-25T20:55:00"                            │
│       }                                                                 │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 8: SAVE CONSENT HISTORY (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ ArtifactPathBuilder + ResultsRepository                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ • save_consent_history(case_id, history)                                │
│   ├─ Path: artifacts/CASE-2025-001/consent/                             │
│   ├─ Save to: history.json                                              │
│   └─ Content:                                                           │
│       [                                                                 │
│         {                                                               │
│           "timestamp": "2025-11-25T20:51:00",                           │
│           "action": "SESSION_CREATED",                                  │
│           "details": "Consent session created"                          │
│         },                                                              │
│         {                                                               │
│           "timestamp": "2025-11-25T20:52:00",                           │
│           "action": "LINK_SENT",                                        │
│           "details": "Approval link sent to nominee"                    │
│         },                                                              │
│         {                                                               │
│           "timestamp": "2025-11-25T20:55:00",                           │
│           "action": "APPROVED",                                         │
│           "details": "Consent approved by nominee"                      │
│         }                                                               │
│       ]                                                                 │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 9: SYNC TO DATABASE (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ InstantApprovalSync (modules/consent/models.py)                         │
├──────────────────────────────────────────────────────────────────────────┤
│ • sync_to_database()                                                    │
│   ├─ Connection: Online ✅                                              │
│   ├─ Sync immediately (no queue needed)                                 │
│   ├─ Store in PostgreSQL:                                               │
│   │   ├─ consent_sessions table                                         │
│   │   ├─ consent_approvals table                                        │
│   │   └─ consent_audit_trail table                                      │
│   ├─ Sync Status: "SYNCED"                                              │
│   └─ Sync Time: 2025-11-25 20:55:05                                     │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 10: NOTIFY USER (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ NotificationHandler (modules/consent/models.py)                         │
├──────────────────────────────────────────────────────────────────────────┤
│ • Send confirmation:                                                    │
│   ├─ Email: "Consent approved for case CASE-2025-001"                   │
│   ├─ SMS: "Consent approved" (optional)                                 │
│   └─ Timestamp: 2025-11-25 20:55:10                                     │
│                                                                          │
│ Status: ONLINE ✅                                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 11: READY FOR EXTRACTION (Online)
┌──────────────────────────────────────────────────────────────────────────┐
│ Extraction Module (modules/extraction/)                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ • Consent Status: APPROVED ✅                                           │
│ • Can proceed with extraction                                           │
│ • Load consent from artifacts:                                          │
│   └─ artifacts/CASE-2025-001/consent/sessions.json                      │
│                                                                          │
│ Status: READY FOR EXTRACTION ✅                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 ONLINE CONSENT ARTIFACT STRUCTURE

```
artifacts/CASE-2025-001/consent/
├── sessions.json
│   └─ Contains: Session details, status, timestamps
│
├── approvals.json
│   └─ Contains: Approval records, timestamps, approver info
│
└── history.json
    └─ Contains: Timeline of all consent actions
```

---

## ✅ ONLINE CONSENT WORKFLOW SUMMARY

| Step | Action | Location | Status |
|------|--------|----------|--------|
| 1 | User initiates consent | Streamlit UI | Online ✅ |
| 2 | Create consent session | ConsentManager | Online ✅ |
| 3 | Save to artifacts | sessions.json | Online ✅ |
| 4 | Generate approval link | ApprovalLinkGenerator | Online ✅ |
| 5 | Nominee approves | Web link | Online ✅ |
| 6 | Save approval record | approvals.json | Online ✅ |
| 7 | Update session status | sessions.json | Online ✅ |
| 8 | Save consent history | history.json | Online ✅ |
| 9 | Sync to database | PostgreSQL | Online ✅ |
| 10 | Notify user | Email/SMS | Online ✅ |
| 11 | Ready for extraction | Extraction Module | Online ✅ |

---

## 🔄 DATA PERSISTENCE

**Artifacts (Local):**
- ✅ artifacts/CASE-2025-001/consent/sessions.json
- ✅ artifacts/CASE-2025-001/consent/approvals.json
- ✅ artifacts/CASE-2025-001/consent/history.json

**Database (PostgreSQL):**
- ✅ consent_sessions table
- ✅ consent_approvals table
- ✅ consent_audit_trail table

**Sync Status:** SYNCED ✅

---

## ✅ KEY FEATURES - ONLINE WORKFLOW

1. **Immediate Sync**: No queuing needed (online)
2. **Dual Storage**: Artifacts + Database
3. **Audit Trail**: Complete history
4. **Notifications**: Real-time updates
5. **Approval Links**: Secure, one-time use
6. **Status Tracking**: PENDING → APPROVED
7. **Error Handling**: Graceful fallback
8. **Logging**: All actions logged

---

## 🚀 READY FOR EXTRACTION

Once consent is approved and synced:
- ✅ Extraction can proceed
- ✅ All consent data available locally (artifacts)
- ✅ All consent data available in database
- ✅ Audit trail complete
- ✅ Nominee notified

**Status**: COMPLETE ✅
