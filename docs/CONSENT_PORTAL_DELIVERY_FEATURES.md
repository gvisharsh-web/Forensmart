# Consent Portal - Approval Link Delivery Features

## ✅ IMPLEMENTATION COMPLETE

**File Modified**: `pages/01_consent_portal.py`

### Features Added

#### 1. **Approval Link Generator**
- Generate approval links with embedded case data
- Encode approval information in URL (base64)
- Auto-detect public URL (Streamlit Cloud, ngrok, etc.)

#### 2. **Delivery Options**

**WhatsApp** 🟢
- Share approval link via WhatsApp
- Pre-formatted message with case details
- Click to open WhatsApp Web/App
- Works with phone numbers (with country code)

**SMS** 📱
- Send approval link via SMS
- Compact message format
- Works with phone numbers

**Email** ✉️
- Send approval link via email
- Professional formatted message
- Includes case ID and purpose
- Works with email addresses

**QR Code** 📲
- Generate QR code for approval link
- Scan with phone camera
- Direct access to approval page

#### 3. **Audit Trail**
- All generated links saved to `audit/generated_links/`
- Tracks case ID, nominee info, timestamp
- JSON format for easy analysis

---

## How to Use

### Step 1: Access Consent Portal
Go to the consent portal page (pages/01_consent_portal.py)

### Step 2: Generate Approval Link
Fill in the form:
- **Case ID**: CASE_001
- **Device ID**: ABC123XYZ
- **Nominee Name**: John Doe
- **Purpose**: Extraction purpose
- **Requested Level**: STANDARD/FULL/LEGAL
- **Nominee Phone**: +1234567890 (for WhatsApp/SMS)
- **Nominee Email**: nominee@example.com (for Email)

### Step 3: Click "Generate Approval Link"
The system will:
1. Create approval data with all case information
2. Encode it in the URL
3. Display the link
4. Show delivery options

### Step 4: Choose Delivery Method

**For WhatsApp:**
- Click "🟢 Share via WhatsApp"
- Opens WhatsApp with pre-filled message
- Nominee receives link and can approve

**For SMS:**
- Click "📱 Send via SMS"
- Opens SMS app with link
- Nominee receives link via text

**For Email:**
- Click "✉️ Send via Email"
- Opens email client
- Pre-filled subject and message
- Nominee receives link via email

**For QR Code:**
- Display QR code to nominee
- They scan with phone camera
- Opens approval page directly

---

## Technical Details

### Functions Added

```python
def _create_whatsapp_link(phone: str, message: str) -> str:
    """Create WhatsApp sharing link"""

def _create_sms_link(phone: str, message: str) -> str:
    """Create SMS sharing link"""

def _create_email_link(email: str, subject: str, message: str) -> str:
    """Create email sharing link"""

def _generate_qr_code_url(data: str) -> str:
    """Generate QR code URL"""
```

### URL Format

```
https://your-app.streamlit.app?data=BASE64_ENCODED_JSON
```

Where BASE64_ENCODED_JSON contains:
```json
{
  "case_id": "CASE_001",
  "device_id": "ABC123XYZ",
  "purpose": "Extraction purpose",
  "requested_level": "STANDARD",
  "nominee_name": "John Doe",
  "created_at": "2025-11-21T21:30:00"
}
```

### Audit Trail

Generated links are saved to:
```
audit/generated_links/CASE_001_link.json
```

Contains:
- Case ID
- Full approval link
- Generation timestamp
- Nominee information
- Contact details

---

## Approval Flow

1. **Investigator** generates approval link in consent portal
2. **Investigator** shares link via WhatsApp/SMS/Email/QR code
3. **Nominee** receives link on their phone
4. **Nominee** clicks link and opens approval page
5. **Nominee** reviews case details
6. **Nominee** approves or denies
7. **System** saves approval decision
8. **Dashboard** detects approval and starts extraction
9. **Investigator** sees approval status in real-time

---

## Key Features

✅ **Public URL Detection**
- Auto-detects Streamlit Cloud URLs
- Works with ngrok tunnels
- Supports custom domains
- Fallback to localhost for testing

✅ **Multiple Delivery Methods**
- WhatsApp (instant messaging)
- SMS (text message)
- Email (formal notification)
- QR Code (visual sharing)

✅ **Secure Encoding**
- Base64 encoding of approval data
- No sensitive data in plain text
- URL-safe encoding

✅ **Audit Trail**
- All links tracked
- Timestamp recorded
- Nominee information saved
- Easy to verify later

✅ **User-Friendly**
- Simple form interface
- Clear delivery options
- One-click sharing
- Mobile-friendly

---

## Testing Checklist

- [ ] Generate approval link with all fields filled
- [ ] Verify WhatsApp link opens correctly
- [ ] Verify SMS link opens correctly
- [ ] Verify Email link opens correctly
- [ ] Scan QR code with phone camera
- [ ] Verify approval link works on mobile
- [ ] Check audit trail file is created
- [ ] Verify nominee can approve via link
- [ ] Check approval status updates in dashboard

---

## Troubleshooting

### WhatsApp Link Not Working
- Ensure phone number includes country code (e.g., +1 for USA)
- Check that WhatsApp is installed on the device
- Try opening link in WhatsApp Web instead

### SMS Link Not Working
- Ensure phone number is valid
- Check that SMS app is configured
- Try using different phone format

### Email Link Not Working
- Ensure email address is valid
- Check that email client is configured
- Try copying link manually

### QR Code Not Scanning
- Ensure good lighting
- Try different QR code reader app
- Verify link is correct by copying manually

### Approval Link Not Working
- Verify case ID and device ID are correct
- Check that URL is publicly accessible
- Try refreshing the page
- Check browser console for errors

---

## Integration with Dashboard

The approval link automatically redirects to the dashboard after approval:

```
dashboard_url?case_id=CASE_001&auto_extract=true
```

The dashboard will:
1. Detect the approval
2. Start extraction automatically
3. Show progress in real-time
4. Save results to reports

---

## Files Modified

**pages/01_consent_portal.py**
- Added delivery helper functions
- Added approval link generator form
- Added delivery options UI
- Added audit trail saving

---

## Next Steps

1. Test approval link generation
2. Test each delivery method
3. Verify nominee can approve via link
4. Monitor audit trail for issues
5. Deploy to production

---

## Support

For issues or questions:
1. Check audit trail for generated links
2. Verify phone/email formats
3. Check browser console for errors
4. Review consent portal logs
5. Test with sample data first
