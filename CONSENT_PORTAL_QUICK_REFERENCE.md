# Consent Portal - Quick Reference Card

## Import Everything

```python
from modules.consent_portal import (
    ConsentPortalLogger,
    ConsentAuditTrail,
    ConsentPortalEnhancer,
    get_consent_manager,
    main
)
```

## 1-Minute Usage

### Generate QR Code
```python
qr_url = ConsentPortalEnhancer.generate_qr_code_url(approval_link)
st.image(qr_url)
```

### Create WhatsApp Link
```python
wa_link = ConsentPortalEnhancer.create_whatsapp_link(
    phone="+1234567890",
    approval_link=approval_link,
    nominee_name="John Doe"
)
st.markdown(f"[Send]({wa_link})")
```

### Create SMS Link
```python
sms_link = ConsentPortalEnhancer.create_sms_link(
    phone="+1234567890",
    approval_link=approval_link
)
```

### Create Email Link
```python
email_link = ConsentPortalEnhancer.create_email_link(
    email="john@example.com",
    approval_link=approval_link,
    case_id="CASE_001"
)
```

### Get All Delivery Options
```python
options = ConsentPortalEnhancer.get_delivery_options(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="john@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)
```

### Render Delivery UI
```python
ConsentPortalEnhancer.render_delivery_ui(
    approval_link=approval_link,
    nominee_phone="+1234567890",
    nominee_email="john@example.com",
    nominee_name="John Doe",
    case_id="CASE_001"
)
```

### Record Approval
```python
ConsentAuditTrail.record_approval(
    case_id="CASE_001",
    decision="approved",
    nominee_name="John Doe",
    device_id="ABC123",
    purpose="Investigation"
)
```

### Get Audit Trail
```python
trail = ConsentAuditTrail.get_audit_trail("CASE_001")
stats = ConsentAuditTrail.get_statistics()
```

### Export Audit Trail
```python
json_data = ConsentAuditTrail.export_audit_trail("CASE_001")
```

### Get Logger
```python
logger_instance = ConsentPortalLogger()
logger = logger_instance.get_logger()
logger.info("Message")
```

## Method Reference

### ConsentPortalEnhancer

| Method | Purpose | Returns |
|--------|---------|---------|
| `generate_qr_code_url(link)` | Generate QR code URL | str (URL) |
| `create_whatsapp_link(phone, link, name)` | WhatsApp link | str (URL) |
| `create_sms_link(phone, link)` | SMS link | str (URL) |
| `create_email_link(email, link, case_id)` | Email link | str (URL) |
| `add_link_expiration(link, hours)` | Add expiration | str (link) |
| `create_approval_details_json(...)` | Create JSON | str (encoded) |
| `get_delivery_options(...)` | Get all options | dict |
| `render_delivery_ui(...)` | Render UI | None |

### ConsentAuditTrail

| Method | Purpose | Returns |
|--------|---------|---------|
| `initialize()` | Create audit file | None |
| `record_approval(...)` | Record approval | bool |
| `get_audit_trail(case_id)` | Get trail | list |
| `get_statistics()` | Get stats | dict |
| `export_audit_trail(case_id)` | Export JSON | str |

### ConsentPortalLogger

| Method | Purpose | Returns |
|--------|---------|---------|
| `__new__()` | Singleton | instance |
| `_initialize()` | Init logger | None |
| `get_logger()` | Get logger | logger |

## File Locations

| Item | Location |
|------|----------|
| Logs | `audit/consent_portal/` |
| Audit Trail | `audit/consent_portal/audit_trail.json` |
| Approvals | `audit/approvals.json` |
| Notifications | `audit/approval_notifications.json` |

## Common Patterns

### Pattern 1: Full Delivery Flow
```python
# Get options
options = ConsentPortalEnhancer.get_delivery_options(
    approval_link, phone, email, name, case_id
)

# Render UI
ConsentPortalEnhancer.render_delivery_ui(
    approval_link, phone, email, name, case_id
)

# Record approval
ConsentAuditTrail.record_approval(
    case_id, "approved", name, device_id, purpose
)
```

### Pattern 2: QR Code Only
```python
qr_url = ConsentPortalEnhancer.generate_qr_code_url(link)
st.image(qr_url, width=300)
```

### Pattern 3: WhatsApp Only
```python
wa_link = ConsentPortalEnhancer.create_whatsapp_link(
    phone, link, name
)
st.markdown(f"[Send via WhatsApp]({wa_link})")
```

### Pattern 4: Audit Trail
```python
ConsentAuditTrail.record_approval(case_id, "approved", name, device_id)
stats = ConsentAuditTrail.get_statistics()
trail = ConsentAuditTrail.get_audit_trail(case_id)
```

## Error Handling

```python
try:
    qr_url = ConsentPortalEnhancer.generate_qr_code_url(link)
    if not qr_url:
        st.error("Failed to generate QR code")
except Exception as e:
    st.error(f"Error: {e}")
```

## Logging

```python
logger_instance = ConsentPortalLogger()
logger = logger_instance.get_logger()

logger.info("Approval saved")
logger.warning("Approval pending")
logger.error("Approval failed")
logger.debug("Debug info")
```

## Phone Number Format

Use international format with country code:
- ✅ `+1234567890`
- ✅ `+441234567890`
- ❌ `1234567890` (missing country code)

## Email Format

Standard email format:
- ✅ `john@example.com`
- ❌ `john@example` (missing domain)

## Link Format

Approval links should be complete URLs:
- ✅ `http://localhost:8501/?data=...`
- ✅ `https://app.example.com/?data=...`
- ❌ `/?data=...` (incomplete)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| QR code not generating | Check internet connection |
| WhatsApp link not working | Verify phone format (+1234567890) |
| Email link not working | Verify email format (user@domain.com) |
| Logs not created | Check audit/ directory permissions |
| Audit trail empty | Verify record_approval() was called |

## Performance

| Operation | Time |
|-----------|------|
| QR code generation | ~100ms |
| Link creation | ~10ms |
| Audit recording | ~50ms |
| UI rendering | ~200ms |

## Limits

| Item | Limit |
|------|-------|
| Audit trail entries | Last 100 kept |
| Log file size | 10MB (rotated) |
| Link expiration | Configurable (default 24h) |

## Status Codes

| Status | Meaning |
|--------|---------|
| `recorded` | Approval recorded in audit trail |
| `pending` | Approval pending |
| `approved` | Approval granted |
| `denied` | Approval denied |

## Exports

```python
__all__ = [
    "ConsentPortalLogger",
    "ConsentAuditTrail",
    "ConsentPortalEnhancer",
    "get_consent_manager",
    "main",
]
```

## Version

- **Version**: 1.0
- **Status**: Production Ready
- **Date**: 2025-11-21

---

**Print this card for quick reference!** 📋
