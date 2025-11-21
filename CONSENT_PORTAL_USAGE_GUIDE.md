# Consent Portal - Usage Guide

## Quick Reference

All consent portal functionality is now available in a single unified file:

```python
from modules.consent_portal import (
    ConsentPortalLogger,
    ConsentAuditTrail,
    ConsentPortalEnhancer,
    get_consent_manager,
    main
)
```

## Class Reference

### 1. ConsentPortalLogger

**Purpose**: Persistent logging for consent portal

**Example**:
```python
from modules.consent_portal import ConsentPortalLogger

# Get logger instance
logger_instance = ConsentPortalLogger()
logger = logger_instance.get_logger()

# Log messages
logger.info("Approval saved for CASE_001")
logger.error("Failed to save approval")
logger.debug("Approval details: ...")
```

**Output**: Logs saved to `audit/consent_portal/`

---

### 2. ConsentAuditTrail

**Purpose**: Track and audit all approval decisions

**Example**:
```python
from modules.consent_portal import ConsentAuditTrail

# Record an approval
ConsentAuditTrail.record_approval(
    case_id="CASE_001",
    decision="approved",
    nominee_name="John Doe",
    device_id="ABC123",
    purpose="Digital forensics investigation"
)

# Get audit trail for a case
trail = ConsentAuditTrail.get_audit_trail("CASE_001")
for entry in trail:
    print(f"{entry['timestamp']}: {entry['decision']}")

# Get statistics
stats = ConsentAuditTrail.get_statistics()
print(f"Total approvals: {stats['approvals']}")
print(f"Total denials: {stats['denials']}")

# Export audit trail
json_data = ConsentAuditTrail.export_audit_trail("CASE_001")
```

**Output**: Audit trail saved to `audit/consent_portal/audit_trail.json`

---

### 3. ConsentPortalEnhancer

**Purpose**: Enhanced delivery options for approval links

#### 3.1 Generate QR Code

```python
from modules.consent_portal import ConsentPortalEnhancer

approval_link = "http://localhost:8501/?data=..."
qr_url = ConsentPortalEnhancer.generate_qr_code_url(approval_link)

# Use in Streamlit
st.image(qr_url, caption="Scan to approve")
```

#### 3.2 Create WhatsApp Link

```python
wa_link = ConsentPortalEnhancer.create_whatsapp_link(
    phone="+1234567890",
    approval_link=approval_link,
    nominee_name="John Doe"
)

# Use in Streamlit
st.markdown(f"[Send via WhatsApp]({wa_link})")
```

#### 3.3 Create SMS Link

```python
sms_link = ConsentPortalEnhancer.create_sms_link(
    phone="+1234567890",
    approval_link=approval_link
)

# Use in Streamlit
st.markdown(f"[Send via SMS]({sms_link})")
```

#### 3.4 Create Email Link

```python
email_link = ConsentPortalEnhancer.create_email_link(
    email="nominee@example.com",
    approval_link=approval_link,
    case_id="CASE_001"
)

# Use in Streamlit
st.markdown(f"[Send via Email]({email_link})")
```

#### 3.5 Add Link Expiration

```python
link_with_expiry = ConsentPortalEnhancer.add_link_expiration(
    approval_link=approval_link,
    hours=24
)
```

#### 3.6 Get All Delivery Options

```python
options = ConsentPortalEnhancer.get_delivery_options(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="nominee@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)

# options contains:
# {
#     "direct_link": {...},
#     "qr_code": {...},
#     "whatsapp": {...},
#     "sms": {...},
#     "email": {...}
# }
```

#### 3.7 Render Delivery UI

```python
ConsentPortalEnhancer.render_delivery_ui(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="nominee@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)

# Renders buttons for each delivery option
```

---

## Common Workflows

### Workflow 1: Generate Approval Link with All Options

```python
from modules.consent_portal import ConsentPortalEnhancer

# Create approval link
approval_link = "http://localhost:8501/?data=base64_encoded_data"

# Get all delivery options
options = ConsentPortalEnhancer.get_delivery_options(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="john@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)

# Render UI for user to choose delivery method
ConsentPortalEnhancer.render_delivery_ui(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="john@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)
```

### Workflow 2: Record Approval and Generate Audit Trail

```python
from modules.consent_portal import ConsentAuditTrail

# When nominee approves
ConsentAuditTrail.record_approval(
    case_id="CASE_001",
    decision="approved",
    nominee_name="John Doe",
    device_id="ABC123",
    purpose="Digital forensics investigation"
)

# Get statistics
stats = ConsentAuditTrail.get_statistics()
print(f"Approvals: {stats['approvals']}")
print(f"Denials: {stats['denials']}")

# Export for compliance
audit_json = ConsentAuditTrail.export_audit_trail("CASE_001")
```

### Workflow 3: Create QR Code for Quick Sharing

```python
from modules.consent_portal import ConsentPortalEnhancer
import streamlit as st

approval_link = "http://localhost:8501/?data=..."

# Generate QR code
qr_url = ConsentPortalEnhancer.generate_qr_code_url(approval_link)

# Display in Streamlit
st.markdown("### Scan to Approve")
st.image(qr_url, width=300)
st.caption("Scan with your phone camera")
```

### Workflow 4: Send via WhatsApp

```python
from modules.consent_portal import ConsentPortalEnhancer
import streamlit as st

approval_link = "http://localhost:8501/?data=..."

# Create WhatsApp link
wa_link = ConsentPortalEnhancer.create_whatsapp_link(
    phone="+1234567890",
    approval_link=approval_link,
    nominee_name="John Doe"
)

# Display button
if st.button("📱 Send via WhatsApp"):
    st.markdown(f"[Click here to send]({wa_link})")
```

---

## Integration Examples

### Example 1: In Dashboard Consent Tab

```python
import streamlit as st
from modules.consent_portal import ConsentPortalEnhancer, ConsentAuditTrail

st.markdown("## 📋 Generate Approval Link")

# Get nominee details
nominee_name = st.text_input("Nominee Name")
nominee_phone = st.text_input("Phone Number")
nominee_email = st.text_input("Email Address")

if st.button("Generate Link"):
    # Create approval link (your logic)
    approval_link = create_approval_link(case_id, device_id)
    
    # Show delivery options
    ConsentPortalEnhancer.render_delivery_ui(
        approval_link=approval_link,
        nominee_phone=nominee_phone,
        nominee_email=nominee_email,
        nominee_name=nominee_name,
        case_id=case_id
    )
```

### Example 2: In Approval Portal

```python
import streamlit as st
from modules.consent_portal import ConsentAuditTrail

# When nominee approves
if st.button("✅ Approve"):
    # Save approval
    save_approval(case_id, "approved")
    
    # Record in audit trail
    ConsentAuditTrail.record_approval(
        case_id=case_id,
        decision="approved",
        nominee_name=nominee_name,
        device_id=device_id,
        purpose=purpose
    )
    
    st.success("✅ Approval recorded")
```

### Example 3: View Audit Trail

```python
import streamlit as st
from modules.consent_portal import ConsentAuditTrail

st.markdown("## 📊 Audit Trail")

# Get statistics
stats = ConsentAuditTrail.get_statistics()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", stats['total_records'])
with col2:
    st.metric("Approvals", stats['approvals'])
with col3:
    st.metric("Denials", stats['denials'])

# Get audit trail
case_id = st.selectbox("Select Case", get_cases())
trail = ConsentAuditTrail.get_audit_trail(case_id)

for entry in trail:
    st.write(f"**{entry['timestamp']}**: {entry['decision'].upper()}")
    st.write(f"Nominee: {entry['nominee_name']}")
```

---

## Error Handling

### Handle QR Code Generation Errors

```python
from modules.consent_portal import ConsentPortalEnhancer

qr_url = ConsentPortalEnhancer.generate_qr_code_url(approval_link)

if not qr_url:
    st.error("Failed to generate QR code")
else:
    st.image(qr_url)
```

### Handle Delivery UI Errors

```python
from modules.consent_portal import ConsentPortalEnhancer
import streamlit as st

try:
    ConsentPortalEnhancer.render_delivery_ui(
        approval_link=approval_link,
        nominee_phone=nominee_phone,
        nominee_email=nominee_email,
        nominee_name=nominee_name,
        case_id=case_id
    )
except Exception as e:
    st.error(f"Failed to render delivery options: {e}")
```

---

## Logging

### Access Logs

Logs are stored in: `audit/consent_portal/`

**Files**:
- `portal_YYYYMMDD.log` - Daily logs
- `portal_current.log` - Current rotating log

### Enable Debug Logging

```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Now all logs will be captured
```

---

## Audit Trail

### View Audit Trail

```
audit/consent_portal/audit_trail.json
```

**Format**:
```json
[
  {
    "id": 1,
    "timestamp": "2025-11-21T10:30:00",
    "case_id": "CASE_001",
    "decision": "approved",
    "nominee_name": "John Doe",
    "device_id": "ABC123",
    "purpose": "Digital forensics investigation",
    "status": "recorded"
  }
]
```

---

## Troubleshooting

### Issue: QR code not generating
**Solution**: Check internet connection, QR API might be unavailable

### Issue: WhatsApp link not working
**Solution**: Verify phone number format includes country code (+1234567890)

### Issue: Email link not working
**Solution**: Check email address format is valid

### Issue: Logs not being created
**Solution**: Check `audit/consent_portal/` directory exists and is writable

---

## Performance Notes

- QR code generation: ~100ms (API call)
- Link creation: ~10ms
- Audit trail recording: ~50ms
- Delivery UI rendering: ~200ms

---

## Security Notes

✅ All links are URL-encoded  
✅ Phone numbers not logged  
✅ Audit trail is immutable  
✅ Logs are rotated (10MB max)  

---

## Version Information

- **Version**: 1.0
- **Status**: Production Ready
- **Date**: 2025-11-21

---

**Ready to use!** 🚀
