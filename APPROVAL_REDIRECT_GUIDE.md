# Approval Redirect & Auto-Extraction Feature Guide

## Overview

This feature implements automatic recognition of approvals and triggers extraction without manual intervention. When a nominee approves an extraction request, the system:

1. **Saves the approval** to the approval database
2. **Notifies the dashboard** via the notification system
3. **Redirects the nominee** back to the dashboard
4. **Automatically starts extraction** if configured

## Components Created

### 1. `modules/approval_redirect.py`
Handles redirect links and approval notifications.

**Key Classes:**
- `ApprovalRedirect`: Creates redirect links and manages redirect configurations
- `ApprovalNotifier`: Notifies dashboard of approvals and manages notifications

**Key Methods:**
```python
# Create a redirect link that triggers extraction
link = ApprovalRedirect.create_redirect_link(
    base_url="http://localhost:8501",
    case_id="CASE_001",
    device_id="ABC123",
    redirect_to="extraction",
    extraction_type="android"
)

# Notify dashboard of approval
ApprovalNotifier.notify_approval(
    case_id="CASE_001",
    device_id="ABC123",
    decision="approved",
    nominee_name="John Doe",
    extraction_type="android"
)

# Get pending notifications
notifications = ApprovalNotifier.get_pending_notifications()
```

### 2. `modules/approval_auto_extraction.py`
Handles automatic extraction triggering when approval is received.

**Key Classes:**
- `ApprovalAutoExtraction`: Checks approval status and triggers extraction

**Key Methods:**
```python
# Check if approval exists and trigger extraction
result = ApprovalAutoExtraction.check_and_trigger_extraction(
    case_id="CASE_001",
    device_id="ABC123",
    extraction_type="android"
)

# Get auto-extraction parameters from URL
params = ApprovalAutoExtraction.get_auto_extraction_params()

# Render auto-extraction UI
should_start = ApprovalAutoExtraction.render_auto_extraction_ui(
    case_id="CASE_001",
    device_id="ABC123"
)
```

### 3. Updated `modules/consent_portal.py`
- Imports the new redirect and notification modules
- Sends approval notifications when nominee approves/denies
- Redirects nominee back to dashboard after approval

## How It Works

### Approval Flow

```
1. Investigator creates approval link in dashboard
   ↓
2. Nominee receives link (via WhatsApp, SMS, Email, etc.)
   ↓
3. Nominee clicks link → Consent Portal opens
   ↓
4. Nominee reviews details and clicks "Approve"
   ↓
5. Consent Portal:
   - Saves approval to database
   - Notifies dashboard via ApprovalNotifier
   - Shows redirect message
   - Redirects to dashboard with auto_extract=true
   ↓
6. Dashboard:
   - Detects auto_extract parameter
   - Checks approval status
   - Automatically starts extraction
   - Shows progress in real-time
```

### Redirect Mechanism

When approval is granted, the consent portal redirects using:
```html
<meta http-equiv="refresh" content="0; url=/?case_id=CASE_001&auto_extract=true" />
```

This redirects the nominee back to the dashboard with parameters that trigger auto-extraction.

### Notification System

Approvals are saved to `audit/approval_notifications.json`:
```json
[
  {
    "id": 1700000000000,
    "timestamp": "2025-11-21T10:30:00",
    "case_id": "CASE_001",
    "device_id": "ABC123",
    "decision": "approved",
    "nominee_name": "John Doe",
    "extraction_type": "android",
    "status": "pending",
    "auto_extract": true
  }
]
```

## Integration Steps

### Step 1: Update Dashboard to Check for Auto-Extraction

Add this to your dashboard's main function (e.g., `dashboard.py` or `dashboard_modern.py`):

```python
from modules.approval_auto_extraction import ApprovalAutoExtraction

# At the start of your main() function
auto_extract_params = ApprovalAutoExtraction.get_auto_extraction_params()

if auto_extract_params:
    case_id = auto_extract_params['case_id']
    device_id = auto_extract_params['device_id']
    extraction_type = auto_extract_params['extraction_type']
    
    st.info(f"🔄 Auto-extraction triggered for case {case_id}")
    
    # Check if approval exists
    result = ApprovalAutoExtraction.check_and_trigger_extraction(
        case_id, device_id, extraction_type
    )
    
    if result["triggered"]:
        # Automatically start extraction
        st.session_state['start_extraction'] = True
        st.session_state['extraction_type'] = extraction_type
        st.session_state['case_id'] = case_id
        st.session_state['device_id'] = device_id
```

### Step 2: Update Extraction UI to Handle Auto-Start

In your extraction tab (e.g., `extraction_ui.py`):

```python
from modules.approval_auto_extraction import ApprovalAutoExtraction

# Check if auto-extraction should start
if ApprovalAutoExtraction.should_auto_extract():
    st.session_state['start_extraction'] = True
    st.rerun()
```

### Step 3: Monitor Notifications (Optional)

Add a notification monitor to your dashboard sidebar:

```python
from modules.approval_redirect import ApprovalNotifier

with st.sidebar:
    st.markdown("### 🔔 Approval Notifications")
    
    notifications = ApprovalNotifier.get_pending_notifications()
    
    if notifications:
        st.warning(f"⏳ {len(notifications)} pending approval(s)")
        
        for notif in notifications:
            with st.expander(f"📬 {notif['case_id']} - {notif['decision'].upper()}"):
                st.write(f"**Nominee:** {notif['nominee_name']}")
                st.write(f"**Device:** {notif['device_id']}")
                st.write(f"**Time:** {notif['timestamp']}")
                
                if st.button("✅ Acknowledge", key=f"ack_{notif['id']}"):
                    ApprovalNotifier.acknowledge_notification(notif['id'])
                    st.rerun()
    else:
        st.success("✅ No pending approvals")
```

## Usage Examples

### Example 1: Basic Auto-Extraction

```python
from modules.approval_auto_extraction import ApprovalAutoExtraction

# In your dashboard main function
params = ApprovalAutoExtraction.get_auto_extraction_params()

if params:
    case_id = params['case_id']
    device_id = params['device_id']
    
    result = ApprovalAutoExtraction.check_and_trigger_extraction(
        case_id, device_id
    )
    
    if result["triggered"]:
        st.success("Starting extraction automatically...")
        # Start extraction here
```

### Example 2: Create Approval Link with Redirect

```python
from modules.approval_redirect import ApprovalRedirect

# Create a redirect link
redirect_link = ApprovalRedirect.create_approval_listener_url(
    base_dashboard_url="http://localhost:8501",
    case_id="CASE_001",
    device_id="ABC123",
    extraction_type="android"
)

# Save configuration
ApprovalRedirect.save_redirect_config(
    case_id="CASE_001",
    config={
        "device_id": "ABC123",
        "extraction_type": "android",
        "redirect_link": redirect_link,
        "nominee_name": "John Doe"
    }
)
```

### Example 3: Monitor Approvals

```python
from modules.approval_redirect import ApprovalNotifier

# Get all pending notifications
pending = ApprovalNotifier.get_pending_notifications()

for notification in pending:
    print(f"Case {notification['case_id']}: {notification['decision']}")
    
    # Acknowledge when processed
    ApprovalNotifier.acknowledge_notification(notification['id'])
```

## File Structure

```
audit/
├── approval_notifications.json    # Approval notifications
└── redirects/
    └── CASE_001_redirect.json     # Redirect configuration
```

## Testing

### Test 1: Manual Approval

1. Create a case in the dashboard
2. Generate an approval link
3. Open the approval link in a browser
4. Click "Approve"
5. Verify redirect happens automatically
6. Check that extraction starts on dashboard

### Test 2: Check Notifications

```python
from modules.approval_redirect import ApprovalNotifier

# View pending notifications
notifications = ApprovalNotifier.get_pending_notifications()
print(f"Pending: {len(notifications)}")

# View all notifications
all_notifications = ApprovalNotifier.get_pending_notifications()
for n in all_notifications:
    print(f"{n['case_id']}: {n['decision']}")
```

### Test 3: Auto-Extraction

1. Create a case
2. Generate approval link
3. Approve via consent portal
4. Verify auto-extraction starts automatically
5. Check progress in real-time

## Troubleshooting

### Issue: Redirect not working

**Solution:**
- Check that `unsafe_allow_html=True` is enabled in Streamlit
- Verify the redirect URL is correct
- Check browser console for errors

### Issue: Approval not recognized

**Solution:**
- Verify approval was saved: Check `audit/approvals.json`
- Clear cache: `ApprovalSync.clear_cache(case_id)`
- Check approval status: `ApprovalSync.get_approval_status(case_id, use_cache=False)`

### Issue: Auto-extraction not starting

**Solution:**
- Verify URL parameters: Check `st.query_params`
- Check approval status: `ApprovalAutoExtraction.check_and_trigger_extraction()`
- Verify extraction UI is checking for auto-extract flag

## Security Considerations

1. **Approval Validation**: Always verify approval status before starting extraction
2. **Redirect URLs**: Use HTTPS in production
3. **Notification Storage**: Notifications are stored locally; implement encryption for sensitive data
4. **Access Control**: Ensure only authorized users can trigger extraction

## Performance Notes

- Notifications are kept to last 100 entries for performance
- Cache TTL is 5 minutes (configurable in `ApprovalSync`)
- Redirect is immediate (no server-side delay)
- Auto-extraction check is lightweight

## Future Enhancements

1. **Email Notifications**: Notify investigator when approval is received
2. **Webhook Support**: Send approval notifications to external systems
3. **Approval Expiration**: Auto-expire approvals after X hours
4. **Approval History**: Track all approval decisions with timestamps
5. **Bulk Approvals**: Handle multiple cases in one approval session

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in the modules
3. Check the audit logs in `audit/` directory
4. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

**Version:** 1.0  
**Last Updated:** 2025-11-21  
**Status:** Production Ready
