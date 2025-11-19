# ⚡ CONSENT PORTAL LOGGING - QUICK IMPLEMENTATION GUIDE

**Date**: 2025-11-19 16:56 UTC+05:30  
**Time to Implement**: 90 minutes  
**Difficulty**: MEDIUM

---

## 🎯 WHAT'S THE PROBLEM?

Consent portal doesn't save logging. Every time it opens, logs are lost.

**Why?**
- No file handler configured
- Logger created on-the-fly without persistence
- Streamlit caching prevents re-initialization
- No audit trail implementation

---

## 🛠️ 3-STEP QUICK FIX

### **STEP 1: Create Logging Module** (20 min)

**File**: `modules/consent_portal_logger.py`

```python
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json

class ConsentPortalLogger:
    """Persistent logging for consent portal."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize logger with file handlers."""
        self._logger = logging.getLogger('consent_portal')
        self._logger.setLevel(logging.DEBUG)
        
        # Create audit directory
        audit_dir = Path('audit/consent_portal')
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler (text log)
        log_file = audit_dir / f'portal_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
        
        # Rotating file handler
        rotating_handler = logging.handlers.RotatingFileHandler(
            audit_dir / 'portal_current.log',
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(file_formatter)
        self._logger.addHandler(rotating_handler)
    
    def get_logger(self):
        """Get the configured logger."""
        return self._logger
    
    @staticmethod
    def log_approval(case_id: str, decision: str, nominee_name: str):
        """Log approval decision."""
        logger = ConsentPortalLogger().get_logger()
        logger.info(f"Approval: {decision.upper()} | Case: {case_id} | Nominee: {nominee_name}")
```

---

### **STEP 2: Create Audit Trail Module** (15 min)

**File**: `modules/consent_audit_trail.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class ConsentAuditTrail:
    """Structured audit trail for approvals."""
    
    AUDIT_FILE = Path('audit/consent_portal/audit_trail.json')
    
    @classmethod
    def initialize(cls):
        """Create audit file if needed."""
        cls.AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls.AUDIT_FILE.exists():
            cls.AUDIT_FILE.write_text(json.dumps([], indent=2))
    
    @classmethod
    def record_approval(cls,
                       case_id: str,
                       decision: str,
                       nominee_name: str,
                       device_id: str) -> bool:
        """Record approval to audit trail."""
        try:
            cls.initialize()
            
            # Read existing trail
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            # Create new entry
            entry = {
                'id': len(trail) + 1,
                'timestamp': datetime.now().isoformat(),
                'case_id': case_id,
                'decision': decision,
                'nominee_name': nominee_name,
                'device_id': device_id,
                'status': 'recorded'
            }
            
            trail.append(entry)
            
            # Write back
            cls.AUDIT_FILE.write_text(json.dumps(trail, indent=2))
            return True
        except Exception as e:
            print(f"Failed to record audit trail: {e}")
            return False
    
    @classmethod
    def get_audit_trail(cls, case_id: Optional[str] = None) -> list:
        """Retrieve audit trail."""
        try:
            cls.initialize()
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            if case_id:
                return [e for e in trail if e['case_id'] == case_id]
            return trail
        except Exception:
            return []
    
    @classmethod
    def get_statistics(cls) -> dict:
        """Get audit trail statistics."""
        trail = cls.get_audit_trail()
        
        return {
            'total_records': len(trail),
            'approvals': len([e for e in trail if e['decision'] == 'approved']),
            'denials': len([e for e in trail if e['decision'] == 'denied']),
            'cases': len(set(e['case_id'] for e in trail))
        }
```

---

### **STEP 3: Update Consent Portal** (20 min)

**File**: `modules/consent_portal.py`

**Add at top (after imports)**:
```python
from modules.consent_portal_logger import ConsentPortalLogger
from modules.consent_audit_trail import ConsentAuditTrail
```

**Update _save_approval function** (around line 58):
```python
def _save_approval(case_id: str, decision: str, nominee_name: Optional[str] = None, 
                   message: Optional[str] = None, approval_link: Optional[str] = None) -> bool:
    """Save approval decision with structured logging."""
    try:
        # Use the unified approval_utils to save decision
        success = save_approval_decision(case_id, decision, nominee_name, message)
        
        if success:
            # Also save the approval link separately for tracking
            approvals_file = get_approvals_file()
            approvals = {}
            
            if approvals_file.exists():
                try:
                    approvals = json.loads(approvals_file.read_text())
                except Exception:
                    approvals = {}
            
            # Update with link info
            if case_id in approvals:
                approvals[case_id]['approval_link'] = approval_link
                approvals_file.write_text(json.dumps(approvals, indent=2))
            
            # LOG WITH STRUCTURED FORMAT:
            portal_logger = ConsentPortalLogger()
            portal_logger.log_approval(
                case_id=case_id,
                decision=decision,
                nominee_name=nominee_name or 'Unknown'
            )
            
            # RECORD TO AUDIT TRAIL:
            device_id = approvals.get(case_id, {}).get('device_id', 'UNKNOWN')
            ConsentAuditTrail.record_approval(
                case_id=case_id,
                decision=decision,
                nominee_name=nominee_name or 'Unknown',
                device_id=device_id
            )
            
            st.success(f"✅ Approval saved successfully for case {case_id}")
            st.info(f"📁 Saved to: `{approvals_file}`")
            st.caption(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            st.error(f"Failed to save approval for case {case_id}")
            return False
    except Exception as e:
        portal_logger = ConsentPortalLogger()
        portal_logger.get_logger().error(f"Failed to save approval: {e}", exc_info=True)
        st.error(f"Failed to save approval: {e}")
        return False
```

**Add audit dashboard to main()** (around line 172):
```python
def main() -> None:
    st.set_page_config(page_title="ForenSmart Consent Portal", layout="wide")
    st.markdown("## 🔐 ForenSmart Consent Portal")
    
    # Sidebar for audit trail
    with st.sidebar:
        st.markdown("### 📊 Audit Trail")
        
        stats = ConsentAuditTrail.get_statistics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", stats['total_records'])
        with col2:
            st.metric("Approvals", stats['approvals'])
        with col3:
            st.metric("Denials", stats['denials'])
        
        # View audit trail
        if st.checkbox("View Audit Trail"):
            trail = ConsentAuditTrail.get_audit_trail()
            if trail:
                st.markdown("#### Recent Entries")
                for entry in trail[-10:]:
                    with st.expander(f"{entry['timestamp'][:10]} - {entry['case_id']}"):
                        st.json(entry)
            else:
                st.info("No audit trail records yet")
        
        # Export audit trail
        if st.button("📥 Export Audit Trail"):
            trail = ConsentAuditTrail.get_audit_trail()
            st.download_button(
                label="Download as JSON",
                data=json.dumps(trail, indent=2),
                file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # ... rest of main() function
```

---

## ✅ VERIFICATION CHECKLIST

After implementation:

- [ ] `modules/consent_portal_logger.py` created
- [ ] `modules/consent_audit_trail.py` created
- [ ] `modules/consent_portal.py` updated with imports
- [ ] `_save_approval()` updated with logging
- [ ] Audit dashboard added to sidebar
- [ ] Test: Approve a request
- [ ] Check: `audit/consent_portal/portal_*.log` file created
- [ ] Check: `audit/consent_portal/audit_trail.json` has entry
- [ ] Test: View audit trail in sidebar
- [ ] Test: Export audit trail as JSON

---

## 📁 FILES CREATED

After implementation:

```
audit/consent_portal/
├── portal_20251119.log          ← Daily log file
├── portal_current.log           ← Current rotating log
└── audit_trail.json             ← Structured audit trail
```

---

## 🧪 TEST COMMANDS

```bash
# Check if logging module exists
ls -la modules/consent_portal_logger.py

# Check if audit trail created
cat audit/consent_portal/audit_trail.json

# Check log file
tail -20 audit/consent_portal/portal_current.log

# Count approvals
grep "APPROVED" audit/consent_portal/portal_current.log | wc -l
```

---

## 🎯 EXPECTED OUTPUT

### **Log File** (`portal_20251119.log`)
```
2025-11-19 16:56:30 | INFO     | consent_portal | Approval: APPROVED | Case: CASE_001 | Nominee: John Doe
2025-11-19 16:56:35 | INFO     | consent_portal | Approval: DENIED | Case: CASE_002 | Nominee: Jane Smith
```

### **Audit Trail** (`audit_trail.json`)
```json
[
  {
    "id": 1,
    "timestamp": "2025-11-19T16:56:30.123456",
    "case_id": "CASE_001",
    "decision": "approved",
    "nominee_name": "John Doe",
    "device_id": "ABC123",
    "status": "recorded"
  }
]
```

---

## 🚀 NEXT STEPS

1. Create the 2 new modules
2. Update consent_portal.py
3. Test approval logging
4. Verify files are created
5. Test export functionality
6. Deploy to production

---

## 📞 TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Log file not created | Check permissions on `audit/` directory |
| Audit trail empty | Approve a request to create entry |
| Export button not working | Check JSON format in audit_trail.json |
| Logging not appearing | Verify imports at top of file |

---

**Quick Guide Generated**: 2025-11-19 16:56 UTC+05:30  
**Status**: READY TO IMPLEMENT  
**Time**: 90 minutes
