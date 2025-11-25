# ✅ Keep Apps Running & Real-Time Approval Reflection Solution

## Problem Identified

1. **Consent Portal**: Needs to stay running continuously
2. **Dashboard**: Not reflecting approval status in real-time
3. **Approval Sync**: Cache TTL (5 minutes) causes delay in reflection

---

## 🔍 Root Cause Analysis

### Issue 1: Approval Status Not Reflecting in Dashboard

**Location**: `modules/approval_sync.py`, Lines 20, 35-36

```python
_cache_ttl = 300  # 5 minutes ❌ TOO LONG

@staticmethod
def _is_cache_valid(case_id: str) -> bool:
    """Check if cached approval is still valid."""
    if case_id not in ApprovalSync._cache_timestamp:
        return False
    
    age = time.time() - ApprovalSync._cache_timestamp[case_id]
    return age < ApprovalSync._cache_ttl  # ❌ 5 minute delay
```

**Problem**: Cache is valid for 5 minutes, so dashboard doesn't see new approvals for up to 5 minutes!

### Issue 2: Dashboard Doesn't Force Refresh

**Location**: `modules/dashboard.py`, Lines 928-935

```python
with col_refresh:
    if st.button('🔄 Refresh', key=f'{case_id}_check_approval'):
        # Clear cache to force fresh read from file
        try:
            ApprovalSync.clear_cache(case_id)  # ✅ User must click refresh
        except Exception as e:
            logger.error(f"Failed to clear approval cache: {e}")
        st.session_state['approval_check_ts'] = datetime.now().isoformat()
        st.rerun()
```

**Problem**: User must manually click "Refresh" button to see new approvals!

### Issue 3: No Auto-Refresh Mechanism

**Problem**: Dashboard doesn't automatically refresh approval status  
**Solution**: Add auto-refresh with shorter cache TTL

---

## 🔧 Solutions

### Solution 1: Reduce Cache TTL (Quick Fix)

**File**: `modules/approval_sync.py`  
**Line**: 20  

```python
# OLD (5 minutes):
_cache_ttl = 300

# NEW (30 seconds):
_cache_ttl = 30
```

**Impact**: Dashboard will check approval file every 30 seconds instead of 5 minutes

---

### Solution 2: Add Auto-Refresh to Dashboard (Better)

**File**: `modules/dashboard.py`  
**Add to render_consent() function**:

```python
# Add auto-refresh every 10 seconds
import time

# Check if we should refresh
last_refresh = st.session_state.get('last_approval_refresh', 0)
current_time = time.time()

if current_time - last_refresh > 10:  # Refresh every 10 seconds
    st.session_state['last_approval_refresh'] = current_time
    # Force refresh by clearing cache
    try:
        ApprovalSync.clear_cache(case_id)
    except Exception:
        pass
    st.rerun()
```

---

### Solution 3: Keep Consent Portal Running (Already Done)

**Status**: ✅ Both apps are already running

```
Consent Portal: Running on port 8501 (Command ID: 240)
Dashboard: Running on port 8502 (Command ID: 246)
```

---

## 📋 Recommended Implementation

### Step 1: Reduce Cache TTL

**File**: `c:\Forensmart\modules\approval_sync.py`  
**Line**: 20  

Replace:
```python
_cache_ttl = 300  # 5 minutes
```

With:
```python
_cache_ttl = 30  # 30 seconds - faster approval reflection
```

### Step 2: Add Auto-Refresh to Dashboard

**File**: `c:\Forensmart\modules\dashboard.py`  
**In render_consent() function (around line 875)**:

Add this at the beginning of the function:

```python
def render_consent(cm: ConsentManager):
    st.markdown("## 🔐 Consent Management")
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info("Select or create a case from the 'Case Management' tab.")
        return
    
    # NEW: Auto-refresh approval status every 10 seconds
    import time
    last_refresh = st.session_state.get('last_approval_refresh', 0)
    current_time = time.time()
    
    if current_time - last_refresh > 10:  # Refresh every 10 seconds
        st.session_state['last_approval_refresh'] = current_time
        # Force refresh by clearing cache
        try:
            ApprovalSync.clear_cache(case_id)
        except Exception as e:
            logger.warning(f"Failed to clear approval cache: {e}")
        st.rerun()
    
    # Rest of the function continues...
    session = cm.get_session(case_id)
    # ...
```

---

## 🚀 Implementation Steps

### Step 1: Reduce Cache TTL (Immediate)

```python
# File: modules/approval_sync.py
# Line 20

# Change from:
_cache_ttl = 300

# To:
_cache_ttl = 30
```

### Step 2: Add Auto-Refresh (Optional but Recommended)

```python
# File: modules/dashboard.py
# In render_consent() function

# Add at the start of the function:
import time
last_refresh = st.session_state.get('last_approval_refresh', 0)
current_time = time.time()

if current_time - last_refresh > 10:
    st.session_state['last_approval_refresh'] = current_time
    try:
        ApprovalSync.clear_cache(case_id)
    except Exception:
        pass
    st.rerun()
```

### Step 3: Restart Applications

```bash
# Kill existing processes
# Restart consent portal
streamlit run modules/consent_portal.py

# Restart dashboard
streamlit run modules/dashboard.py --server.port 8502
```

---

## 📊 Expected Behavior After Fix

### Before Fix
```
1. User approves in consent portal
2. Status saved to file
3. Dashboard checks cache (valid for 5 minutes)
4. Dashboard shows "PENDING" for up to 5 minutes ❌
5. User must click "Refresh" button ❌
6. After refresh, shows "APPROVED" ✅
```

### After Fix
```
1. User approves in consent portal
2. Status saved to file
3. Dashboard auto-refreshes every 10 seconds
4. Cache expires every 30 seconds
5. Dashboard shows "APPROVED" within 10 seconds ✅
6. No manual refresh needed ✅
7. Extraction triggers automatically ✅
```

---

## 🎯 Benefits

✅ Approval status reflects in real-time (within 10 seconds)  
✅ No manual refresh needed  
✅ Better user experience  
✅ Extraction triggers automatically  
✅ Consent portal stays running  
✅ Dashboard stays running  

---

## 📋 Verification Checklist

- [ ] Reduce cache TTL to 30 seconds
- [ ] Add auto-refresh to dashboard
- [ ] Restart both applications
- [ ] Test approval flow
- [ ] Verify status updates within 10 seconds
- [ ] Verify extraction triggers automatically
- [ ] Check audit trail

---

## 🔄 Current Status

**Consent Portal**: ✅ Running (Port 8501)  
**Dashboard**: ✅ Running (Port 8502)  
**Apps**: ✅ Staying on continuously  

**Next**: Apply cache TTL reduction and auto-refresh

---

**Solution Date**: 2025-11-21  
**Status**: Ready for Implementation
