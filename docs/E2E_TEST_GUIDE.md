# End-to-End Testing Guide
## Full Approval & Extraction Flow

### Prerequisites
- ✅ Dashboard running on port 8501
- ✅ Network IP detected (e.g., 192.168.1.100)
- ✅ Android phone connected via ADB (optional, for SMS testing)

---

## Test D: Complete End-to-End Flow

### Scenario: Investigator creates case → Nominee approves → Extraction starts

---

### Step 1: Start Dashboard
```bash
streamlit run modules/dashboard_merged.py
```

Expected output:
```
Local URL: http://localhost:8501
Network URL: http://192.168.1.100:8501
```

---

### Step 2: Create a Test Case

1. Open dashboard: http://192.168.1.100:8501
2. Go to **Case Management** tab
3. Click **Create New Case**
4. Fill in:
   - Subject Name: "Test Subject"
   - Device ID: "TEST_DEVICE_001"
   - Consent Level: "STANDARD"
5. Click **Create Case**

Expected result:
- ✅ Case created with ID like `CASE-Test_Subject-20251122190000`
- ✅ Case appears in sidebar

---

### Step 3: Generate Approval Link & Hash

1. Go to **Consent Hub** tab
2. Fill in:
   - Nominee Name: "Test Nominee"
   - Nominee Phone: "+919876543210" (or your test phone)
   - Nominee Email: "test@example.com"
   - Purpose: "Test Extraction"
3. Click **Generate Approval Link & Fallback Hash**

Expected result:
- ✅ Approval link generated (with Network IP)
- ✅ Fallback hash generated (8-char uppercase)
- ✅ Sharing options displayed (QR, WhatsApp, SMS, Email)

Example link:
```
http://192.168.1.100:8501?data=BASE64_ENCODED_DATA
```

Example hash:
```
A7B9C1D2
```

---

### Step 4A: Test Approval via Link (Primary Method)

#### On Nominee's Phone/Browser:

1. Open the approval link on another device/browser:
   ```
   http://192.168.1.100:8501?data=...
   ```

2. You should see:
   - ✅ ONLY approval form (no dashboard tabs)
   - ✅ Case information (Case ID, Device ID, Purpose)
   - ✅ Approve/Deny buttons

3. Click **✅ Yes, Approve**

Expected result:
- ✅ Success message: "Approval Granted!"
- ✅ Balloons animation
- ✅ "This page will close automatically in 3 seconds..."
- ✅ Page auto-closes after 3 seconds

#### On Investigator's Dashboard:

1. Stay on **Consent Hub** tab
2. Watch the **Live Approval Status** section

Expected result:
- ✅ Status updates to: "✅ Approved by Test Nominee at [timestamp]"
- ✅ Balloons animation
- ✅ Message: "🚀 Extraction will start automatically when nominee is redirected to dashboard."

---

### Step 4B: Test Approval via SMS Hash (Fallback Method)

#### Option 1: Manual Entry (No ADB needed)

1. In **Consent Hub**, go to **SMS Hash Fallback** section
2. Left column: **Manual Entry**
3. Copy the fallback hash (e.g., `A7B9C1D2`)
4. Paste it in the text field
5. Click **Verify Hash and Start Extraction**

Expected result:
- ✅ Hash verified
- ✅ Success message: "✅ SMS Fallback Approval successful! Extraction triggered."
- ✅ Extraction starts

#### Option 2: Auto-Read from ADB (If phone connected)

1. In **Consent Hub**, go to **SMS Hash Fallback** section
2. Right column: **Auto-Read from ADB**
3. Verify: "✅ ADB Device Connected"
4. On the connected Android phone, send SMS:
   ```
   APPROVE A7B9C1D2
   ```
5. Click **🔍 Read SMS from Nominee Phone**

Expected result:
- ✅ System reads SMS from phone
- ✅ Hash extracted: `A7B9C1D2`
- ✅ Hash verified
- ✅ Success message: "✅ SMS Auto-Read Successful! Hash A7B9C1D2 verified. Extraction triggered."
- ✅ Extraction starts

---

### Step 5: Verify Extraction Starts

After approval (via link or SMS hash):

1. Dashboard should auto-navigate to **Extraction** tab
2. You should see:
   - ✅ Case ID loaded
   - ✅ Device status (Connected/Offline)
   - ✅ Battery and Storage metrics
   - ✅ Extraction type (Android/iOS/HDD)

3. Extraction should start automatically:
   - ✅ Progress bar appears
   - ✅ Real-time progress updates
   - ✅ Artifacts being extracted

Expected result:
- ✅ Extraction starts without manual button click
- ✅ Progress bar shows real-time updates
- ✅ Artifacts count increases

---

### Step 6: Verify Audit Trail

1. Go to **Diagnostics** tab
2. Check **Approval Status** section

Expected result:
- ✅ "Approved: ✅ Yes"
- ✅ "Expired: ✅ No"
- ✅ Approval timestamp visible

---

## Test Scenarios

### Scenario A: Approval via Link (Primary)
- ✅ Link generated with Network IP
- ✅ Nominee sees ONLY approval form
- ✅ Approval saved to session_state
- ✅ Page auto-closes
- ✅ Investigator dashboard detects approval
- ✅ Extraction starts automatically

### Scenario B: Approval via SMS Hash (Manual)
- ✅ Hash generated and displayed
- ✅ Investigator manually enters hash
- ✅ Hash verified
- ✅ Extraction starts automatically

### Scenario C: Approval via SMS Hash (Auto-Read)
- ✅ Nominee sends SMS with hash
- ✅ ADB reads SMS automatically
- ✅ Hash extracted and verified
- ✅ Extraction starts automatically

### Scenario D: Denial Flow
- ✅ Nominee clicks "❌ No, Deny"
- ✅ Denial saved
- ✅ Page auto-closes
- ✅ Investigator sees denial status

---

## Troubleshooting

### Issue: Link shows localhost instead of Network IP
**Solution:** Check `_get_dashboard_url()` function
```python
# Should return: http://192.168.1.100:8501
# Not: http://localhost:8501
```

### Issue: Nominee sees full dashboard instead of approval form
**Solution:** Check URL parameters
```python
# Correct: http://192.168.1.100:8501?data=BASE64
# Router should detect 'data' param and show consent_view only
```

### Issue: Extraction doesn't start after approval
**Solution:** Check session_state
```python
# Verify: st.session_state['auto_extract_triggered'] = True
# Verify: render_extraction_tab() is called
```

### Issue: SMS reading returns no SMS
**Solution:** Check ADB connection
```bash
adb devices  # Should show connected device
adb shell content query --uri content://sms/inbox  # Should show SMS
```

---

## Success Criteria

✅ All tests pass if:
1. Link generated with correct Network IP
2. Nominee sees ONLY approval form
3. Approval saved and detected by dashboard
4. Extraction starts automatically
5. Progress bar shows real-time updates
6. SMS hash fallback works (manual or auto)
7. Audit trail records all events
8. Page auto-closes after approval

---

## Next Steps

After successful testing:
1. ✅ Fix any issues found
2. ✅ Test on different networks
3. ✅ Test error scenarios (device offline, extraction fails, etc.)
4. ✅ Deploy to production
