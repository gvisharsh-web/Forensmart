# ✅ HYBRID ARCHITECTURE (ONLINE + OFFLINE) - COMPLETE

**Status**: HYBRID SUPPORT ADDED TO CONSENT & EXTRACTION
**Date**: November 25, 2025

---

## 🎯 HYBRID ARCHITECTURE OVERVIEW

**Hybrid Approach**: Online + Offline support with automatic sync

**Benefits:**
- ✅ Works offline without internet
- ✅ Automatic sync when online
- ✅ Local caching for fast access
- ✅ Queue operations for later sync
- ✅ Graceful degradation
- ✅ No data loss

---

## 📋 CONSENT MODULE - HYBRID SUPPORT

### New Class: HybridConnectivityManager

**Features:**
- Connectivity status tracking (online/offline)
- Pending sync queue
- Sync interval management
- Operation queuing

**Methods:**
```python
set_online(is_online)           # Set connectivity status
is_connected()                  # Check if online
queue_for_sync(operation)       # Queue operation
get_pending_sync()              # Get pending operations
mark_synced(index)              # Mark as synced
should_sync()                   # Check if should sync
sync_completed()                # Mark sync complete
```

### ConsentManager Hybrid Methods

**New Methods:**
```python
get_session(case_id)            # Hybrid: local cache + remote
sync_with_remote(remote_url)    # Sync pending operations
queue_operation_offline(...)    # Queue for sync
```

**Hybrid Flow:**
```
1. Check local cache (offline support)
2. Check main sessions
3. Cache locally for future use
4. Queue operations when offline
5. Sync when online
```

### Usage Example:

```python
consent_manager = get_consent_manager()

# Offline: Works with local cache
session = consent_manager.get_session('CASE-001')

# Queue operation if offline
if not consent_manager.connectivity_manager.is_connected():
    consent_manager.queue_operation_offline(
        'create_session',
        {'case_id': 'CASE-001', 'level': 'LEGAL'}
    )

# Sync when online
if consent_manager.connectivity_manager.is_connected():
    consent_manager.sync_with_remote()
```

---

## 📋 EXTRACTION MODULE - HYBRID SUPPORT

### New Class: ExtractionHybridManager

**Features:**
- Connectivity status tracking
- Pending extraction queue
- Sync interval management
- Extraction queuing

**Methods:**
```python
set_online(is_online)           # Set connectivity status
is_connected()                  # Check if online
queue_extraction(id, data)      # Queue extraction
get_pending_extractions()       # Get pending
mark_synced(id)                 # Mark as synced
should_sync()                   # Check if should sync
sync_completed()                # Mark sync complete
```

### ExtractionOrchestrator Hybrid Methods

**New Methods:**
```python
set_connectivity(is_online)     # Set connectivity status
sync_extraction_results()       # Sync with remote
queue_extraction_offline(...)   # Queue for sync
get_results_hybrid(case_id)     # Hybrid results retrieval
get_pending_sync_extractions()  # Get pending
```

**Hybrid Flow:**
```
1. Check local cache (offline support)
2. Check main results
3. Check file storage
4. Cache locally for future use
5. Queue extractions when offline
6. Sync when online
```

### Usage Example:

```python
orchestrator = get_orchestrator()

# Set connectivity status
orchestrator.set_connectivity(is_online=True)

# Offline: Works with local cache
results = orchestrator.get_results_hybrid('CASE-001')

# Queue extraction if offline
if not orchestrator.hybrid_manager.is_connected():
    orchestrator.queue_extraction_offline(
        'CASE-001',
        {'modules': ['communications', 'location']}
    )

# Sync when online
if orchestrator.hybrid_manager.is_connected():
    orchestrator.sync_extraction_results()

# Get pending sync extractions
pending = orchestrator.get_pending_sync_extractions()
```

---

## 🔧 CONFIGURATION

```env
# Hybrid Architecture
REMOTE_SYNC_ENABLED=true
SYNC_INTERVAL_SECONDS=300
EXTRACTION_SYNC_INTERVAL=300
```

---

## 📊 HYBRID ARCHITECTURE COMPONENTS

### Consent Module

| Component | Purpose |
|-----------|---------|
| HybridConnectivityManager | Manage connectivity & sync queue |
| local_cache | Store sessions locally |
| connectivity_manager | Track online/offline status |
| sync_with_remote() | Sync pending operations |
| queue_operation_offline() | Queue operations |

### Extraction Module

| Component | Purpose |
|-----------|---------|
| ExtractionHybridManager | Manage connectivity & sync queue |
| local_results_cache | Store results locally |
| hybrid_manager | Track online/offline status |
| sync_extraction_results() | Sync pending extractions |
| queue_extraction_offline() | Queue extractions |
| get_results_hybrid() | Hybrid results retrieval |

---

## 🔄 SYNC WORKFLOW

### Online to Offline Transition
```
1. User goes offline
2. Operations queued locally
3. Local cache used for reads
4. No data loss
```

### Offline to Online Transition
```
1. User comes online
2. Check if sync needed
3. Sync pending operations
4. Mark as synced
5. Clear queue
```

### Automatic Sync
```
1. Check connectivity
2. Check sync interval
3. Get pending operations
4. Sync with remote
5. Mark as synced
6. Update last sync time
```

---

## 📈 BENEFITS

✅ **Offline Support**: Works without internet
✅ **Automatic Sync**: Syncs when online
✅ **Local Caching**: Fast access to recent data
✅ **No Data Loss**: All operations queued
✅ **Graceful Degradation**: Degrades gracefully
✅ **Transparent**: Works seamlessly
✅ **Configurable**: Sync intervals configurable
✅ **Scalable**: Works for any number of operations

---

## 🎯 HYBRID ARCHITECTURE FLOW

```
┌─────────────────────────────────────────────────────┐
│         FORENSMART HYBRID ARCHITECTURE              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         CONNECTIVITY MANAGER                 │  │
│  │  - Online/Offline Status                     │  │
│  │  - Sync Queue Management                     │  │
│  │  - Sync Interval Tracking                    │  │
│  └──────────────────────────────────────────────┘  │
│                      │                              │
│         ┌────────────┼────────────┐                │
│         │            │            │                │
│    ┌────▼────┐  ┌───▼────┐  ┌───▼────┐           │
│    │ ONLINE  │  │OFFLINE │  │ SYNCING│           │
│    │ MODE    │  │ MODE   │  │ MODE   │           │
│    └────┬────┘  └───┬────┘  └───┬────┘           │
│         │           │            │                │
│    ┌────▼───────────▼────────────▼────┐           │
│    │     LOCAL CACHE + QUEUE           │           │
│    │  - Session Cache                  │           │
│    │  - Results Cache                  │           │
│    │  - Pending Operations Queue       │           │
│    └───────────────────────────────────┘           │
│                                                     │
│    ┌───────────────────────────────────┐           │
│    │     REMOTE SERVER (CLOUD)         │           │
│    │  - Persistent Storage             │           │
│    │  - Central Sync Point             │           │
│    └───────────────────────────────────┘           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## ✅ HYBRID ARCHITECTURE COMPLETE

Status: READY FOR PHASE 3

Hybrid Features Implemented:
- ✅ Online/Offline connectivity tracking
- ✅ Local caching for offline support
- ✅ Operation queuing for sync
- ✅ Automatic sync when online
- ✅ Sync interval management
- ✅ Graceful degradation
- ✅ No data loss
- ✅ Transparent to users

---

## 📁 FILES UPDATED

- ✅ `modules/consent/models.py` - Hybrid support added
- ✅ `modules/extraction/orchestrator.py` - Hybrid support added

---

## 🚀 READY FOR PHASE 3

All hybrid architecture features complete with:
- ✅ Consent hybrid support
- ✅ Extraction hybrid support
- ✅ Online/offline detection
- ✅ Local caching
- ✅ Operation queuing
- ✅ Automatic sync
- ✅ Graceful degradation
