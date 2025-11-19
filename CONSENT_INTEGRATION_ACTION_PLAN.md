# 🎯 CONSENT INTEGRATION - ACTION PLAN

**Date**: 2025-11-19 16:50 UTC+05:30  
**Priority**: CRITICAL  
**Estimated Time**: 2-3 hours  
**Status**: READY FOR IMPLEMENTATION

---

## 📋 QUICK SUMMARY

**Problem**: Consent portal approvals are NOT reflecting in the dashboard.

**Root Cause**: 6 critical integration gaps:
1. Approval status saved to file, but dashboard reads stale cache
2. Device detected separately in portal and dashboard
3. Approval links saved but not retrievable
4. Consent level never updated after approval
5. Delivery UI method doesn't exist
6. Cache invalidation requires manual refresh

**Solution**: Synchronize data flow through `ConsentSession` object.

---

## 🔧 STEP-BY-STEP FIXES

### **STEP 1: Add Approval Fields to ConsentSession** (15 min)

**File**: `modules/consent.py` lines 600-650

**Current**:
```python
@dataclass
class ConsentSession:
    case_id: str
    device_id: Optional[str] = None
    level: ConsentLevel = ConsentLevel.NONE
    nominee_phone: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_verified: Optional[str] = None
    consent_history: List[Tuple[datetime, ConsentLevel, str]] = field(default_factory=list)
    sms_attempts: int = 0
    primary_evidence: bool = False
```

**Required Changes**:
```python
@dataclass
class ConsentSession:
    case_id: str
    device_id: Optional[str] = None
    level: ConsentLevel = ConsentLevel.NONE
    nominee_phone: Optional[str] = None
    
    # ADD THESE FIELDS:
    approval_status: Optional[str] = None  # 'pending', 'approved', 'denied'
    approval_timestamp: Optional[str] = None
    approval_link: Optional[str] = None
    nominee_name: Optional[str] = None
    # END NEW FIELDS
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_verified: Optional[str] = None
    consent_history: List[Tuple[datetime, ConsentLevel, str]] = field(default_factory=list)
    sms_attempts: int = 0
    primary_evidence: bool = False
```

**Checklist**:
- [ ] Add `approval_status` field
- [ ] Add `approval_timestamp` field
- [ ] Add `approval_link` field
- [ ] Add `nominee_name` field
- [ ] Update `_write_consent_snapshot()` to include new fields
- [ ] Update `_read_consent_snapshot()` to read new fields
- [ ] Test persistence

---

### **STEP 2: Update Consent Portal to Sync Approval** (20 min)

**File**: `modules/consent_portal.py` lines 281-298

**Current**:
```python
if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
    if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
        _save_approval_link(case_id, str(st.query_params), nominee_name)
        try:
            from modules.approval_sync import ApprovalSync
            ApprovalSync.clear_cache(case_id)
        except Exception:
            pass
        st.success("✅ **Approval Granted**")
```

**Required Changes**:
```python
if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
    if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
        _save_approval_link(case_id, str(st.query_params), nominee_name)
        
        # ADD THIS BLOCK:
        # Update ConsentSession with approval status
        cm = get_consent_manager()
        session = cm.get_session(case_id)
        if session:
            session.approval_status = 'approved'
            session.approval_timestamp = datetime.now().isoformat()
            session.nominee_name = nominee_name
            session.approval_link = str(st.query_params)
            cm.persist_session(case_id)
        # END NEW BLOCK
        
        try:
            from modules.approval_sync import ApprovalSync
            ApprovalSync.clear_cache(case_id)
        except Exception:
            pass
        st.success("✅ **Approval Granted**")
```

**Checklist**:
- [ ] Get ConsentManager instance
- [ ] Get session for case_id
- [ ] Update approval_status to 'approved'
- [ ] Update approval_timestamp to current time
- [ ] Update nominee_name
- [ ] Update approval_link
- [ ] Call persist_session()
- [ ] Test approval saves to session

---

### **STEP 3: Update Dashboard to Read from Session** (25 min)

**File**: `modules/dashboard.py` lines 875-990

**Current**:
```python
def render_consent(cm: ConsentManager):
    st.markdown("## 🔐 Consent Management")
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info("Select or create a case from the 'Case Management' tab.")
        return
    session = cm.get_session(case_id)
    if not session:
        st.warning("No consent session found for this case.")
        return

    unlock_status = {}
    unlock_fn = getattr(cm, 'get_unlock_status', None)
    if callable(unlock_fn):
        unlock_status = unlock_fn(case_id)
    
    # Check for approval decision from consent portal with ApprovalSync
    approval_decision = get_approval_decision(case_id)
    
    # Use ApprovalSync for real-time approval status
    if ApprovalSync.is_approved(case_id):
        approval_decision = 'approved'
    elif ApprovalSync.is_denied(case_id):
        approval_decision = 'denied'
```

**Required Changes**:
```python
def render_consent(cm: ConsentManager):
    st.markdown("## 🔐 Consent Management")
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info("Select or create a case from the 'Case Management' tab.")
        return
    session = cm.get_session(case_id)
    if not session:
        st.warning("No consent session found for this case.")
        return

    # CHANGE THIS BLOCK:
    # Read approval status from ConsentSession (source of truth)
    approval_decision = session.approval_status or 'pending'
    approval_timestamp = session.approval_timestamp
    # END CHANGED BLOCK
    
    # Keep unlock_status for backward compatibility
    unlock_status = {}
    unlock_fn = getattr(cm, 'get_unlock_status', None)
    if callable(unlock_fn):
        unlock_status = unlock_fn(case_id)
```

**Also update display section** (around line 920):
```python
# CHANGE THIS:
col_approval, col_refresh = st.columns([3, 1])
with col_approval:
    if approval_decision == 'approved':
        st.success(f"✅ **Nominee Approved** - Extraction is now unlocked!")
    elif approval_decision == 'denied':
        st.error(f"❌ **Nominee Denied** - Extraction request was rejected.")
    else:
        st.info("⏳ Waiting for nominee approval...")

with col_refresh:
    if st.button('🔄 Refresh', key=f'{case_id}_check_approval'):
        try:
            ApprovalSync.clear_cache(case_id)
        except Exception as e:
            logger.error(f"Failed to clear approval cache: {e}")
        st.session_state['approval_check_ts'] = datetime.now().isoformat()
        st.rerun()

# TO THIS:
col_approval, col_refresh = st.columns([3, 1])
with col_approval:
    if approval_decision == 'approved':
        st.success(f"✅ **Nominee Approved** at {approval_timestamp}")
    elif approval_decision == 'denied':
        st.error(f"❌ **Nominee Denied** at {approval_timestamp}")
    else:
        st.info("⏳ Waiting for nominee approval...")

with col_refresh:
    if st.button('🔄 Refresh', key=f'{case_id}_check_approval'):
        # Reload session from disk
        cm.reload_session(case_id)
        st.session_state['approval_check_ts'] = datetime.now().isoformat()
        st.rerun()
```

**Checklist**:
- [ ] Read approval_status from session
- [ ] Read approval_timestamp from session
- [ ] Update display to show timestamp
- [ ] Add reload_session() call on refresh
- [ ] Remove ApprovalSync cache clearing (no longer needed)
- [ ] Test dashboard shows approval immediately

---

### **STEP 4: Implement render_delivery_ui()** (20 min)

**File**: `modules/consent_portal_enhanced.py` (add new method)

**Add this method** after `add_link_expiration()`:
```python
@staticmethod
def render_delivery_ui(
    approval_link: str,
    nominee_phone: str = "",
    nominee_email: str = "",
    nominee_name: str = "",
    case_id: str = ""
) -> None:
    """Render delivery options UI in Streamlit."""
    try:
        import streamlit as st
        from urllib.parse import quote_plus
    except ImportError:
        logger.error("Streamlit not available for render_delivery_ui")
        return
    
    st.markdown("### 📤 Delivery Options")
    
    col1, col2, col3 = st.columns(3)
    
    # WhatsApp
    with col1:
        st.markdown("**🟢 WhatsApp**")
        if nominee_phone:
            whatsapp_link = ConsentPortalEnhancer.create_whatsapp_link(
                nominee_phone, approval_link, nominee_name
            )
            if whatsapp_link:
                st.markdown(f"[Send via WhatsApp]({whatsapp_link})")
            else:
                st.warning("Invalid phone number")
        else:
            st.info("Enter nominee phone to send via WhatsApp")
    
    # SMS
    with col2:
        st.markdown("**📱 SMS**")
        if nominee_phone:
            sms_link = ConsentPortalEnhancer.create_sms_link(nominee_phone, approval_link)
            if sms_link:
                st.markdown(f"[Send SMS]({sms_link})")
            else:
                st.warning("Invalid phone number")
        else:
            st.info("Enter nominee phone to send SMS")
    
    # Email
    with col3:
        st.markdown("**✉️ Email**")
        if nominee_email:
            email_link = ConsentPortalEnhancer.create_email_link(
                nominee_email, approval_link, case_id
            )
            if email_link:
                st.markdown(f"[Send Email]({email_link})")
            else:
                st.warning("Invalid email address")
        else:
            st.info("Enter nominee email to send via email")
    
    # QR Code
    st.markdown("### 📲 QR Code")
    qr_url = ConsentPortalEnhancer.generate_qr_code_url(approval_link)
    if qr_url:
        st.image(qr_url, caption="Scan to approve", width=200)
    else:
        st.warning("Could not generate QR code")
    
    # Copy link
    st.markdown("### 🔗 Direct Link")
    st.text_input("Copy this link:", value=approval_link, disabled=True)
```

**Checklist**:
- [ ] Add render_delivery_ui() method
- [ ] Add WhatsApp link generation
- [ ] Add SMS link generation
- [ ] Add Email link generation
- [ ] Add QR code display
- [ ] Add direct link copy
- [ ] Test button works in dashboard

---

### **STEP 5: Unify Device Detection** (15 min)

**File**: `modules/consent.py` (add method)

**Add this method** to `ConsentManager` class:
```python
def get_or_detect_device(self, case_id: str) -> Optional[str]:
    """Get device ID from session or detect it."""
    session = self.sessions.get(case_id)
    if not session:
        return None
    
    # If already set, return it
    if session.device_id and session.device_id != 'UNKNOWN_DEVICE':
        return session.device_id
    
    # Otherwise, detect and save
    detected = self.ensure_device_id(case_id)
    if detected:
        session.device_id = detected
        self._write_consent_snapshot(case_id)
    
    return detected
```

**Update consent_portal.py** (lines 244-253):
```python
# CHANGE FROM:
if device_id == 'UNKNOWN_DEVICE' or not device_id:
    try:
        detected = cm.ensure_device_id(case_id)
        if detected:
            device_id = detected

# TO:
if device_id == 'UNKNOWN_DEVICE' or not device_id:
    try:
        detected = cm.get_or_detect_device(case_id)
        if detected:
            device_id = detected
```

**Update dashboard.py** (lines 991):
```python
# CHANGE FROM:
detected_device = cm.ensure_device_id(case_id)

# TO:
detected_device = cm.get_or_detect_device(case_id)
```

**Checklist**:
- [ ] Add get_or_detect_device() method
- [ ] Update portal to use new method
- [ ] Update dashboard to use new method
- [ ] Test device detection consistent

---

### **STEP 6: Add Approval Retrieval Methods** (15 min)

**File**: `modules/consent.py` (add methods)

**Add these methods** to `ConsentManager` class:
```python
def get_approval_history(self, case_id: str) -> List[Dict[str, Any]]:
    """Get approval history for a case."""
    try:
        from modules.approval_utils import get_approvals_file
        
        approvals_file = get_approvals_file()
        if not approvals_file.exists():
            return []
        
        approvals = json.loads(approvals_file.read_text())
        if case_id in approvals:
            return approvals[case_id].get('history', [])
    except Exception as e:
        logger.error(f"Failed to get approval history: {e}")
    
    return []

def get_latest_approval_link(self, case_id: str) -> Optional[str]:
    """Get the latest approval link for a case."""
    session = self.sessions.get(case_id)
    if session and session.approval_link:
        return session.approval_link
    
    # Fallback to file
    try:
        from modules.approval_utils import get_approvals_file
        
        approvals_file = get_approvals_file()
        if approvals_file.exists():
            approvals = json.loads(approvals_file.read_text())
            if case_id in approvals:
                return approvals[case_id].get('approval_link')
    except Exception as e:
        logger.error(f"Failed to get approval link: {e}")
    
    return None
```

**Checklist**:
- [ ] Add get_approval_history() method
- [ ] Add get_latest_approval_link() method
- [ ] Test retrieval works
- [ ] Update dashboard to use new methods

---

## 📊 VERIFICATION CHECKLIST

### **Test 1: Approval Sync** ✓
- [ ] Nominee approves in portal
- [ ] Dashboard shows approval within 2 seconds
- [ ] Approval timestamp displayed
- [ ] No manual refresh needed

### **Test 2: Device Detection** ✓
- [ ] Device detected in portal
- [ ] Dashboard shows same device
- [ ] Device consistent across tabs
- [ ] No separate detection calls

### **Test 3: Approval Link** ✓
- [ ] Link saved after approval
- [ ] Link retrievable from dashboard
- [ ] Link history available
- [ ] Can resend link

### **Test 4: Consent Level** ✓
- [ ] Level updated after approval
- [ ] Dashboard shows updated level
- [ ] Level persisted to disk
- [ ] Level survives app restart

### **Test 5: Delivery Options** ✓
- [ ] Button doesn't crash
- [ ] QR code displays
- [ ] WhatsApp link works
- [ ] SMS link works
- [ ] Email link works

### **Test 6: Multiple Cases** ✓
- [ ] Each case has independent approval
- [ ] No cross-case contamination
- [ ] Device detection per case
- [ ] Approval history per case

---

## 🚀 IMPLEMENTATION TIMELINE

| Step | Task | Time | Status |
|------|------|------|--------|
| 1 | Add fields to ConsentSession | 15 min | ⏳ Pending |
| 2 | Update consent portal sync | 20 min | ⏳ Pending |
| 3 | Update dashboard to read from session | 25 min | ⏳ Pending |
| 4 | Implement render_delivery_ui() | 20 min | ⏳ Pending |
| 5 | Unify device detection | 15 min | ⏳ Pending |
| 6 | Add approval retrieval methods | 15 min | ⏳ Pending |
| **Total** | | **110 min** | |

---

## 📝 TESTING COMMANDS

```bash
# Test 1: Run dashboard
streamlit run modules/dashboard.py

# Test 2: Open consent portal
streamlit run modules/consent_portal.py

# Test 3: Check approval file
cat audit/approvals.json

# Test 4: Check consent snapshot
cat case_snapshots/CASE_001/consent_session.json

# Test 5: Run tests
pytest tests/ -v
```

---

## 🔗 RELATED FILES

- `modules/consent.py` - Core consent logic
- `modules/consent_portal.py` - Approval portal
- `modules/dashboard.py` - Main UI
- `modules/consent_portal_enhanced.py` - Delivery options
- `modules/approval_utils.py` - Approval file management
- `modules/approval_sync.py` - Approval synchronization

---

## 📞 SUPPORT

If you encounter issues:

1. Check `CONSENT_INTEGRATION_ERROR_REPORT.md` for detailed analysis
2. Check `CONSENT_PORTAL_DASHBOARD_LINE_COMPARISON.md` for line-by-line comparison
3. Review test results in `Testing Checklist` above
4. Check logs: `app_error_log.txt`

---

**Document Generated**: 2025-11-19 16:50 UTC+05:30  
**Status**: READY FOR IMPLEMENTATION  
**Next Step**: Start with STEP 1
