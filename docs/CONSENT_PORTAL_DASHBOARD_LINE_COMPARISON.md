# 📊 CONSENT PORTAL ↔ DASHBOARD LINE-BY-LINE COMPARISON

**Purpose**: Identify exact misalignments between consent portal and dashboard  
**Date**: 2025-11-19 16:50 UTC+05:30

---

## 🔴 CRITICAL MISALIGNMENTS

### **MISMATCH #1: Approval Decision Flow**

#### Consent Portal (`consent_portal.py` lines 281-298)
```python
281→    col1, col2 = st.columns(2)
282→    with col1:
283→        if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
284→            # Get current URL as the approval link
285→            current_url = st.query_params.get('_url', 'N/A')
286→            if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
287→                # Also save the link separately for tracking
288→                _save_approval_link(case_id, str(st.query_params), nominee_name)
289→                
290→                # Clear cache to ensure dashboard sees the approval immediately
291→                try:
292→                    from modules.approval_sync import ApprovalSync
293→                    ApprovalSync.clear_cache(case_id)
294→                except Exception:
295→                    pass
296→                
297→                st.success("✅ **Approval Granted** - Thank you for your consent.")
298→                st.caption(f"Nominee: {nominee_name or 'Not specified'}")
```

**What it does**:
- Line 286: Saves approval to file via `_save_approval()`
- Line 288: Saves approval link separately
- Line 293: Clears ApprovalSync cache
- **MISSING**: Does NOT update `ConsentSession` object

#### Dashboard (`dashboard.py` lines 891-898)
```python
891→    # Check for approval decision from consent portal with ApprovalSync
892→    approval_decision = get_approval_decision(case_id)
893→    
894→    # Use ApprovalSync for real-time approval status
895→    if ApprovalSync.is_approved(case_id):
896→        approval_decision = 'approved'
897→    elif ApprovalSync.is_denied(case_id):
898→        approval_decision = 'denied'
```

**What it does**:
- Line 892: Reads approval from file via `get_approval_decision()`
- Line 895-898: Checks ApprovalSync cache
- **PROBLEM**: Reads from file/cache, not from `ConsentSession`
- **PROBLEM**: No automatic refresh after portal saves

**The Gap**:
```
Portal saves to:        approval_file.json
                              ↓
Dashboard reads from:   ApprovalSync cache (may be stale)
                              ↓
ConsentSession.level    (NEVER UPDATED)
```

---

### **MISMATCH #2: Device Detection & Synchronization**

#### Consent Portal (`consent_portal.py` lines 244-253)
```python
244→        # Attempt to detect device if not provided or unknown
245→        if device_id == 'UNKNOWN_DEVICE' or not device_id:
246→            try:
247→                detected = cm.ensure_device_id(case_id)
248→                if detected:
249→                    device_id = detected
250→                    st.info(f"✅ Device auto-detected: {device_id}")
251→                else:
252→                    st.warning("⚠️ Could not auto-detect device. Please verify manually.")
253→            except Exception as e:
254→                st.warning(f"⚠️ Device detection failed: {e}")
```

**What it does**:
- Line 247: Calls `cm.ensure_device_id(case_id)` to detect device
- Line 249: Updates local `device_id` variable
- **PROBLEM**: Only updates display, not persistent storage
- **PROBLEM**: No sync back to session

#### Dashboard (`dashboard.py` lines 991-1008)
```python
991→    detected_device = cm.ensure_device_id(case_id)
992→    device_label = cm.get_device_label(detected_device)
993→    st.markdown('#### Device Confirmation')
994→    col_dev1, col_dev2 = st.columns(2)
995→    with col_dev1:
996→        st.metric("Detected Device", device_label)
997→    with col_dev2:
998→        st.metric("Consent Level", session.level.name)
999→    refresh_col1, refresh_col2 = st.columns([1, 3])
1000→   with refresh_col1:
1001→       if st.button('🔄 Refresh device detection', key=f'{case_id}_refresh_device'):
1002→           updated_device = cm.ensure_device_id(case_id)
1003→           st.session_state['device_refresh_ts'] = datetime.now().isoformat()
1004→           if updated_device and updated_device != 'UNKNOWN_DEVICE':
1005→               st.success(f"Detected device: {cm.get_device_label(updated_device)}")
1006→           else:
1007→               st.warning('No authorised device detected.')
1008→           st.rerun()
```

**What it does**:
- Line 991: Calls `cm.ensure_device_id(case_id)` again (separate detection)
- Line 1001-1008: Requires manual refresh button
- **PROBLEM**: Detects device separately from portal
- **PROBLEM**: If device changes, portal and dashboard show different devices

**The Gap**:
```
Portal detects:         Device A (via ensure_device_id)
                              ↓
Dashboard detects:      Device B (via separate ensure_device_id call)
                              ↓
Nominee approves for:   Device A
                              ↓
Extraction runs on:     Device B (WRONG!)
```

---

### **MISMATCH #3: Approval Link Storage & Retrieval**

#### Consent Portal (`consent_portal.py` lines 100-120)
```python
100→def _save_approval_link(case_id: str, approval_link: str, nominee_name: Optional[str] = None) -> bool:
101→    """Save approval link for tracking."""
102→    try:
103→        approvals_file = get_approvals_file()
104→        approvals = {}
104→        
105→        if approvals_file.exists():
106→            try:
107→                approvals = json.loads(approvals_file.read_text())
108→            except Exception:
109→                approvals = {}
110→        
111→        if case_id not in approvals:
112→            approvals[case_id] = {}
113→        
114→        approvals[case_id]['approval_link'] = approval_link
115→        approvals[case_id]['nominee_name'] = nominee_name
116→        approvals[case_id]['timestamp'] = datetime.now().isoformat()
117→        
118→        approvals_file.write_text(json.dumps(approvals, indent=2))
119→        return True
120→    except Exception as e:
121→        logger.error(f"Failed to save approval link: {e}")
122→        return False
```

**What it does**:
- Line 114-116: Saves approval link to file
- **PROBLEM**: No method to retrieve it from dashboard

#### Dashboard (`dashboard.py` lines 1099-1119)
```python
1099→   approval_link = None
1100→   token = None
1101→   if st.button('Generate Approval Link', key=f'{case_id}_generate_link'):
1102→       result = cm.create_unlock_approval(case_id, requested_level, purpose, nominee_name)
1103→       if result.get('status') == 'pending':
1104→           token = result.get('token')
1105→           # Build link with embedded approval data for better UX
1106→           approval_data = {
1107→               'case_id': case_id,
1108→               'device_id': detected_device or 'UNKNOWN_DEVICE',
1109→               'purpose': purpose,
1110→               'requested_level': requested_level.name,
1111→               'nominee_name': nominee_name
1112→           }
1113→           approval_link = _build_approval_link(base_url.strip(), token, approval_data)
1114→           st.session_state['latest_approval_link'] = approval_link
1115→           st.success('Approval request created. Share the link below with the nominee.')
1116→       else:
1117→           st.error(result.get('message', 'Unable to create approval link.'))
1118→
1119→   approval_link = approval_link or st.session_state.get('latest_approval_link')
```

**What it does**:
- Line 1102: Creates approval via `cm.create_unlock_approval()`
- Line 1113: Builds approval link
- Line 1114: Stores in session state (temporary)
- **PROBLEM**: Doesn't retrieve previously saved links from file
- **PROBLEM**: No history of approval links

**The Gap**:
```
Portal saves link to:    approval_file.json
                              ↓
Dashboard reads from:    session_state (temporary, lost on refresh)
                              ↓
No way to retrieve:      Previous approval links
```

---

### **MISMATCH #4: Consent Level Update**

#### Consent Portal (`consent_portal.py` lines 281-298)
```python
283→        if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
284→            if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
285→                # Saves to file
286→                # Does NOT update ConsentSession.level
287→                st.success("✅ **Approval Granted**")
```

**What it does**:
- Line 284: Saves approval status to file
- **MISSING**: Does NOT update `session.level` from `NONE` to `STANDARD`

#### Dashboard (`dashboard.py` lines 998-999)
```python
998→    with col_dev2:
999→        st.metric("Consent Level", session.level.name)
```

**What it does**:
- Line 999: Displays `session.level` (still `NONE` after portal approval!)
- **PROBLEM**: Shows old consent level

**The Gap**:
```
Portal approves:        approval_file.json = 'approved'
                              ↓
ConsentSession.level:   NONE (NEVER UPDATED)
                              ↓
Dashboard shows:        "Consent Level: NONE" (WRONG!)
```

---

### **MISMATCH #5: Delivery UI Integration**

#### Dashboard (`dashboard.py` lines 1126-1134)
```python
1126→       if st.button('📤 Show Delivery Options', key=f'{case_id}_show_delivery'):
1127→           # Render delivery UI with QR code, WhatsApp, SMS, Email options
1128→           ConsentPortalEnhancer.render_delivery_ui(
1129→               approval_link=approval_link,
1130→               nominee_phone=nominee_contact,
1131→               nominee_email=nominee_email,
1132→               nominee_name=nominee_name,
1133→               case_id=case_id
1134→           )
```

**What it does**:
- Line 1128: Calls `ConsentPortalEnhancer.render_delivery_ui()`
- **PROBLEM**: Method doesn't exist!

#### Consent Portal Enhanced (`consent_portal_enhanced.py` lines 1-210)
```python
1→  """Enhanced consent portal with QR codes and link delivery."""
2→  from __future__ import annotations
3→  
4→  import logging
5→  import json
6→  import base64
7→  from typing import Dict, Any, Optional
8→  from urllib.parse import quote
9→  
10→ logger = logging.getLogger(__name__)
11→
12→
13→ class ConsentPortalEnhancer:
14→     """Enhance consent portal with QR codes and delivery options."""
15→
16→     @staticmethod
17→     def generate_qr_code_url(approval_link: str) -> str:
17→         # ... implementation
18→
19→     @staticmethod
20→     def create_whatsapp_link(phone: str, approval_link: str, nominee_name: str = "") -> str:
20→         # ... implementation
21→
22→     @staticmethod
23→     def create_sms_link(phone: str, approval_link: str) -> str:
23→         # ... implementation
24→
25→     @staticmethod
26→     def create_email_link(email: str, approval_link: str, case_id: str = "") -> str:
26→         # ... implementation
27→
28→     # NO render_delivery_ui() METHOD!
```

**What it does**:
- Has static methods for link generation
- **MISSING**: `render_delivery_ui()` method for Streamlit UI

**The Gap**:
```
Dashboard calls:        ConsentPortalEnhancer.render_delivery_ui()
                              ↓
Method doesn't exist:   AttributeError
                              ↓
Button crashes:         "Show Delivery Options" fails
```

---

### **MISMATCH #6: Cache Invalidation Timing**

#### Consent Portal (`consent_portal.py` lines 289-293)
```python
289→                # Clear cache to ensure dashboard sees the approval immediately
290→                try:
291→                    from modules.approval_sync import ApprovalSync
292→                    ApprovalSync.clear_cache(case_id)
293→                except Exception:
294→                    pass
```

**What it does**:
- Line 292: Clears cache AFTER saving approval
- **PROBLEM**: Dashboard may have already cached the old value

#### Dashboard (`dashboard.py` lines 927-935)
```python
927→    with col_refresh:
928→        if st.button('🔄 Refresh', key=f'{case_id}_check_approval'):
929→            # Clear cache to force fresh read from file
930→            try:
931→                ApprovalSync.clear_cache(case_id)
932→            except Exception as e:
933→                logger.error(f"Failed to clear approval cache: {e}")
934→            st.session_state['approval_check_ts'] = datetime.now().isoformat()
935→            st.rerun()
```

**What it does**:
- Line 928-935: Requires MANUAL refresh button
- **PROBLEM**: Nominee approves, but dashboard doesn't update until user clicks refresh

**The Gap**:
```
Portal clears cache:    ApprovalSync.clear_cache()
                              ↓
Dashboard still cached: Old approval status
                              ↓
User must click:        "🔄 Refresh" button
                              ↓
Dashboard updates:      Shows new approval (DELAYED)
```

---

## 📋 SUMMARY TABLE

| Issue | Portal Code | Dashboard Code | Gap | Severity |
|-------|-------------|-----------------|-----|----------|
| Approval sync | `consent_portal.py:286` | `dashboard.py:892` | Portal saves file, dashboard reads cache | CRITICAL |
| Device detection | `consent_portal.py:247` | `dashboard.py:991` | Separate detections, may differ | CRITICAL |
| Approval link storage | `consent_portal.py:114` | `dashboard.py:1114` | Portal saves, dashboard can't retrieve | HIGH |
| Consent level update | `consent_portal.py:284` | `dashboard.py:999` | Portal doesn't update session level | HIGH |
| Delivery UI | `dashboard.py:1128` | `consent_portal_enhanced.py:13` | Method doesn't exist | HIGH |
| Cache invalidation | `consent_portal.py:292` | `dashboard.py:931` | Manual refresh required | MEDIUM |

---

## 🔧 REQUIRED SYNCHRONIZATION POINTS

### **Sync Point #1: Approval Status**
```
consent_portal.py:284  →  Save to file
                       →  Update ConsentSession.approval_status
                       →  Clear ApprovalSync cache
                       →  Trigger dashboard refresh

dashboard.py:892       ←  Read from ConsentSession (not file)
                       ←  Auto-refresh on approval
```

### **Sync Point #2: Device ID**
```
consent_portal.py:247  →  Detect device
                       →  Save to ConsentSession.device_id
                       →  Persist to disk

dashboard.py:991       ←  Read from ConsentSession.device_id
                       ←  No separate detection needed
```

### **Sync Point #3: Approval Link**
```
consent_portal.py:114  →  Save to file
                       →  Save to ConsentSession.approval_link
                       →  Return link to caller

dashboard.py:1114      ←  Read from ConsentSession.approval_link
                       ←  Retrieve history from file
```

### **Sync Point #4: Consent Level**
```
consent_portal.py:284  →  Update ConsentSession.level
                       →  Save to file
                       →  Persist to disk

dashboard.py:999       ←  Read from ConsentSession.level
                       ←  Display updated level
```

---

## 🚀 IMPLEMENTATION ORDER

1. **Step 1**: Add fields to `ConsentSession` (approval_status, approval_timestamp, approval_link)
2. **Step 2**: Update `consent_portal.py` to sync approval to session
3. **Step 3**: Update `dashboard.py` to read from session
4. **Step 4**: Implement `render_delivery_ui()` in `consent_portal_enhanced.py`
5. **Step 5**: Unify device detection
6. **Step 6**: Add approval link retrieval methods

---

**Document Generated**: 2025-11-19 16:50 UTC+05:30  
**Status**: READY FOR IMPLEMENTATION
