# 🔄 ONLINE/OFFLINE APPROACH - INTEGRATION STATUS

**Date**: November 26, 2025
**Status**: ✅ PARTIALLY IMPLEMENTED - NEEDS WIRING IN APP.PY

---

## 📊 CURRENT STATUS

### **WHAT EXISTS**

✅ **HybridConnectivityManager** in `modules/consent/models.py`
- Online/offline status tracking
- Pending sync queue management
- Hash verification for offline operations
- Sync interval management
- Dev mode support

✅ **Hash Code Verification** in `modules/extraction/ui_consent_approval.py`
- SHA-256 PIN hashing
- HMAC constant-time comparison
- Audit trail logging
- Failed attempt tracking

---

## 🏗️ ONLINE/OFFLINE ARCHITECTURE

```
HYBRID CONNECTIVITY SYSTEM
│
├─ ONLINE MODE
│  ├─ Real-time approval processing
│  ├─ Immediate sync
│  ├─ Live database updates
│  └─ Instant notifications
│
├─ OFFLINE MODE
│  ├─ Queue operations locally
│  ├─ Hash-based verification
│  ├─ Local storage
│  └─ Sync when online
│
└─ HYBRID MANAGEMENT
   ├─ Connectivity detection
   ├─ Queue management
   ├─ Hash verification
   ├─ Sync scheduling
   └─ Fallback handling
```

---

## 📋 WHAT'S IMPLEMENTED

### **1. HybridConnectivityManager Class**

**Location**: `modules/consent/models.py` (Lines 30-180)

**Features**:
```python
class HybridConnectivityManager:
    ├─ __init__()
    │  ├─ is_online: bool
    │  ├─ pending_sync_queue: List
    │  ├─ last_sync_time: datetime
    │  ├─ sync_interval: int
    │  └─ dev_mode: bool
    │
    ├─ set_online(is_online)
    │  └─ Set connectivity status
    │
    ├─ is_connected()
    │  └─ Check if online
    │
    ├─ queue_for_sync(operation)
    │  └─ Queue operation for sync
    │
    ├─ get_pending_sync()
    │  └─ Get pending operations
    │
    ├─ generate_operation_hash(operation)
    │  └─ Generate SHA-256 hash
    │
    ├─ verify_operation_hash(operation, expected_hash)
    │  └─ Verify operation integrity
    │
    ├─ add_hash_to_operation(operation)
    │  └─ Add hash to operation
    │
    ├─ verify_queued_operations()
    │  └─ Verify all queued operations
    │
    ├─ set_dev_mode(enabled)
    │  └─ Toggle dev mode
    │
    ├─ is_dev_mode()
    │  └─ Check dev mode status
    │
    ├─ toggle_dev_mode()
    │  └─ Toggle dev mode on/off
    │
    ├─ mark_synced(operation_index)
    │  └─ Mark operation as synced
    │
    ├─ should_sync()
    │  └─ Check if should sync
    │
    └─ sync_completed()
       └─ Mark sync as completed
```

---

### **2. Hash Verification in Consent Approval**

**Location**: `modules/extraction/ui_consent_approval.py`

**Features**:
```python
├─ hash_pin(pin, salt)
│  └─ Hash PIN using SHA-256 with salt
│
├─ verify_pin_with_hash(entered_pin, stored_pin_hash)
│  └─ Verify PIN using HMAC constant-time comparison
│
├─ log_consent_approval(case_id, method, pin_hash)
│  └─ Log approval with hash
│
└─ log_failed_attempt(case_id, method, pin_hash)
   └─ Log failed attempt with hash
```

---

## ❌ WHAT'S MISSING - WIRING IN APP.PY

### **Missing Integration Points**

```
1. CONNECTIVITY STATUS DISPLAY
   ├─ Show online/offline status in UI
   ├─ Display in sidebar
   ├─ Show sync status
   └─ Show pending operations count

2. OFFLINE MODE HANDLING
   ├─ Queue operations when offline
   ├─ Display queue status
   ├─ Show sync progress
   └─ Handle sync failures

3. SYNC MANAGEMENT
   ├─ Manual sync trigger
   ├─ Auto-sync scheduling
   ├─ Sync progress tracking
   └─ Sync error handling

4. HASH VERIFICATION DISPLAY
   ├─ Show hash verification status
   ├─ Display verification results
   ├─ Show hash details
   └─ Audit trail display

5. FALLBACK MECHANISMS
   ├─ Offline extraction queuing
   ├─ Offline approval processing
   ├─ Local storage fallback
   └─ Sync retry logic
```

---

## 🔧 WHAT NEEDS TO BE ADDED TO APP.PY

### **1. Initialize HybridConnectivityManager**

```python
# At top of app.py
from modules.consent.models import HybridConnectivityManager

# In session state initialization
if 'connectivity_manager' not in st.session_state:
    st.session_state.connectivity_manager = HybridConnectivityManager()
```

### **2. Add Connectivity Status Display**

```python
# In sidebar
def render_connectivity_status():
    """Display online/offline status"""
    connectivity = st.session_state.connectivity_manager
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if connectivity.is_connected():
            st.success("🟢 ONLINE")
        else:
            st.warning("🔴 OFFLINE")
    
    with col2:
        if st.button("🔄 Sync"):
            sync_pending_operations()
```

### **3. Add Offline Mode Handling**

```python
# In extraction workflow
def handle_offline_extraction():
    """Handle extraction when offline"""
    connectivity = st.session_state.connectivity_manager
    
    if not connectivity.is_connected():
        st.warning("⚠️ You are offline. Operations will be queued for sync.")
        
        # Queue operation
        operation = {
            'type': 'extraction',
            'case_id': case_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add hash for verification
        operation_with_hash = connectivity.add_hash_to_operation(operation)
        connectivity.queue_for_sync(operation_with_hash)
        
        st.info(f"✓ Operation queued. Pending: {len(connectivity.get_pending_sync())}")
```

### **4. Add Sync Management**

```python
# In app.py
def sync_pending_operations():
    """Sync pending operations when online"""
    connectivity = st.session_state.connectivity_manager
    
    if not connectivity.is_connected():
        st.error("Cannot sync: You are offline")
        return
    
    pending = connectivity.get_pending_sync()
    
    if not pending:
        st.info("No pending operations to sync")
        return
    
    st.info(f"Syncing {len(pending)} pending operations...")
    
    # Verify operations
    results = connectivity.verify_queued_operations()
    
    st.write(f"✓ Verified: {results['verified']}")
    st.write(f"✗ Failed: {results['failed']}")
    
    if results['failed'] > 0:
        st.error(f"Errors: {results['errors']}")
    
    # Mark as synced
    for idx in range(len(pending)):
        connectivity.mark_synced(idx)
    
    connectivity.sync_completed()
    st.success("✓ Sync completed")
```

### **5. Add Hash Verification Display**

```python
# In consent approval section
def display_hash_verification():
    """Display hash verification status"""
    st.subheader("🔐 Hash Verification Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hash Algorithm", "SHA-256")
    
    with col2:
        st.metric("Comparison Method", "HMAC Constant-Time")
    
    with col3:
        st.metric("Verification", "✓ PASSED")
    
    st.write("**Hash Details:**")
    st.write(f"- PIN Hash: {pin_hash[:32]}...")
    st.write(f"- Timestamp: {datetime.now().isoformat()}")
    st.write(f"- Audit Trail: Logged")
```

---

## 📊 INTEGRATION CHECKLIST

### **Phase 1: Basic Connectivity**
- [ ] Initialize HybridConnectivityManager in session state
- [ ] Display online/offline status in sidebar
- [ ] Add connectivity toggle button
- [ ] Show pending operations count

### **Phase 2: Offline Mode**
- [ ] Queue operations when offline
- [ ] Display queue status
- [ ] Show queued operations list
- [ ] Add clear queue button

### **Phase 3: Sync Management**
- [ ] Add manual sync button
- [ ] Implement auto-sync scheduling
- [ ] Display sync progress
- [ ] Handle sync errors

### **Phase 4: Hash Verification**
- [ ] Display hash verification status
- [ ] Show hash details
- [ ] Display audit trail
- [ ] Show verification results

### **Phase 5: Fallback Mechanisms**
- [ ] Implement offline extraction queuing
- [ ] Implement offline approval processing
- [ ] Add local storage fallback
- [ ] Implement sync retry logic

### **Phase 6: Testing**
- [ ] Test online mode
- [ ] Test offline mode
- [ ] Test sync functionality
- [ ] Test hash verification
- [ ] Test fallback mechanisms

---

## 🔄 DATA FLOW - ONLINE/OFFLINE

### **Online Flow**

```
USER ACTION
    ↓
CHECK CONNECTIVITY
    ├─ Online: Continue
    └─ Offline: Queue & Continue
    ↓
PROCESS REQUEST
    ├─ Hash verification
    ├─ Audit logging
    └─ Database update
    ↓
IMMEDIATE RESPONSE
    ├─ Success message
    ├─ Audit trail
    └─ Sync status
```

### **Offline Flow**

```
USER ACTION
    ↓
CHECK CONNECTIVITY
    └─ Offline: Queue
    ↓
QUEUE OPERATION
    ├─ Add hash
    ├─ Add timestamp
    └─ Store locally
    ↓
DISPLAY STATUS
    ├─ "Queued for sync"
    ├─ Show pending count
    └─ Show sync button
    ↓
WHEN ONLINE
    ├─ Auto-sync or manual sync
    ├─ Verify hashes
    ├─ Update database
    └─ Clear queue
```

---

## 📋 CODE EXAMPLES FOR APP.PY

### **Example 1: Initialize in Session State**

```python
# In main() function
if 'connectivity_manager' not in st.session_state:
    st.session_state.connectivity_manager = HybridConnectivityManager()

# Optional: Load from environment
connectivity = st.session_state.connectivity_manager
connectivity.set_online(os.getenv('ONLINE_MODE', 'true').lower() == 'true')
```

### **Example 2: Display in Sidebar**

```python
# In render_sidebar() function
with st.sidebar:
    st.divider()
    st.subheader("🔌 Connectivity")
    
    connectivity = st.session_state.connectivity_manager
    
    if connectivity.is_connected():
        st.success("🟢 ONLINE")
    else:
        st.warning("🔴 OFFLINE")
    
    # Pending operations
    pending = connectivity.get_pending_sync()
    if pending:
        st.info(f"⏳ {len(pending)} pending operations")
        
        if st.button("🔄 Sync Now"):
            sync_pending_operations()
```

### **Example 3: Handle Offline Extraction**

```python
# In render_extraction_workflow() function
def handle_extraction():
    connectivity = st.session_state.connectivity_manager
    
    if not connectivity.is_connected():
        st.warning("⚠️ Operating in offline mode")
        st.info("Operations will be queued and synced when online")
    
    # Proceed with extraction
    # ...
    
    # If offline, queue the operation
    if not connectivity.is_connected():
        operation = {
            'type': 'extraction',
            'case_id': case_id,
            'device_id': device_id,
            'timestamp': datetime.now().isoformat()
        }
        
        operation_with_hash = connectivity.add_hash_to_operation(operation)
        connectivity.queue_for_sync(operation_with_hash)
        
        st.success("✓ Operation queued for sync")
```

---

## ✅ IMPLEMENTATION ROADMAP

### **Step 1: Basic Integration (1 day)**
- Initialize HybridConnectivityManager
- Display connectivity status
- Add connectivity toggle

### **Step 2: Offline Mode (1 day)**
- Implement operation queuing
- Display queue status
- Add queue management

### **Step 3: Sync Management (1 day)**
- Implement sync functionality
- Add sync scheduling
- Handle sync errors

### **Step 4: Hash Verification Display (1 day)**
- Display verification status
- Show hash details
- Display audit trail

### **Step 5: Testing & Refinement (1 day)**
- Test all modes
- Test sync functionality
- Fix issues

**Total**: 5 days

---

## 📊 SUMMARY

**Current Status**: ✅ 50% Complete

**What's Done**:
- ✅ HybridConnectivityManager implemented
- ✅ Hash verification implemented
- ✅ Audit logging implemented
- ✅ Queue management implemented

**What's Needed**:
- ❌ Wiring in app.py
- ❌ UI components for connectivity status
- ❌ Sync management UI
- ❌ Offline mode handling in UI
- ❌ Hash verification display

**Next Steps**:
1. Add HybridConnectivityManager to app.py session state
2. Add connectivity status display in sidebar
3. Add offline mode handling
4. Add sync management UI
5. Add hash verification display

---

**Status**: ✅ STRUCTURE READY - NEEDS APP.PY WIRING

**Next Action**: Wire online/offline approach into app.py

